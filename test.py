# System libs
import os
import argparse
from distutils.version import LooseVersion
import io
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
import skimage
import json
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image, ImageDraw
from tqdm import tqdm
from mit_semseg.config import cfg
import matplotlib.pyplot as plt

colors = loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def filter_labels(labelmap, labels=[]):
    labelmap = labelmap.astype('int')
    height, width = labelmap.shape

    for i in range(height):
        for j in range(width):
            value = labelmap.item((i, j))
            if value not in labels:
                labelmap[i,j] = -1
    return labelmap

def find_boxes(labelmap, target_label=0):
    labelmap = labelmap.astype('int')
    height, width = labelmap.shape

    # box value [left, top, right, bottom] four points
    box = [
        np.array([width, height]),
        np.array([width, height]),
        np.array([0, 0]),
        np.array([0, 0])]
    has_label = False
    for i in range(height):
        for j in range(width):
            value = labelmap.item((i, j))
            if value == target_label:
                has_label = True
                # get left point
                if box[0][0] > j:
                    box[0][0] = j
                    box[0][1] = i
                # # get right point
                if box[2][0] < j:
                    box[2][0] = j
                    box[2][1] = i
                # # get top point
                if box[1][1] > i:
                    box[1][0] = j
                    box[1][1] = i
                # # get bottom point
                if box[3][1] < i:
                    box[3][0] = j
                    box[3][1] = i
    new_bbox = {
        'x1': box[0][0].item(),
        'x2': box[2][0].item(),
        'y1': box[1][1].item(),
        'y2': box[3][1].item()
    }
    return new_bbox if has_label else {}

def visualize_box(image, box, color):
    draw = ImageDraw.Draw(image)
    if len(box) != 0:
        draw.rectangle([(box['x1'], box['y1']), (box['x2'], box['y2'])], outline=color)

def find_max_box(boxes):
    if len(boxes) == 0:
        return {}

    area_boxes = []
    for box in boxes:
        if len(box) != 0:
            area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
        else:
            area = 0
        area_boxes.append(area)
    indexes = np.argsort(area_boxes)[::-1]
    return boxes[indexes[0]]

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_intersection(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    return intersection_area / bb2_area

def visualize_result(data, pred, cfg):
    (img, info) = data
    labels = []
    str_labels = []

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))
            labels.append(uniques[idx])
            str_labels.append(f'{name}, {round(ratio, 2)}\n')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    filename, ext = os.path.splitext(img_name)
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))

    with open(os.path.join(cfg.TEST.result, f"{os.path.splitext(img_name)[0]}.txt"), 'w') as f_out:
        f_out.writelines(str_labels)

    # color table
    fig, ax = plt.subplots(figsize=(4, 4))
    n = len(str_labels)
    ncols = 1
    nrows = n // ncols

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols

    for i, label in enumerate(labels):
        row = i % nrows
        col = i // nrows
        y = Y - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)


        ax.text(xi_text, y, str_labels[i], fontsize=(h * 0.5),
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y + h * 0.1, xi_line, xf_line,
                  color=colors[label]/255, linewidth=(h * 0.6))

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1,
                    top=1, bottom=0,
                    hspace=0, wspace=0)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='jpg')
    buffer.seek(0)
    pil_color_table = Image.open(buffer)

    filename, ext = os.path.splitext(img_name)
    pil_color_table.save(
        os.path.join(cfg.TEST.result, f"{filename}_colortable.jpg")
    )
    buffer.close()

    # [0: wall, 2: sky, 3: floor, 6: road, 13: ground, 15:table]
    target_labels_bg = [0, 2]
    target_labels_base = [3, 6, 13, 15]
    pred_filtered = filter_labels(pred, labels=target_labels_bg+target_labels_base)
    pred_color_filtered = colorEncode(pred_filtered, colors).astype(np.uint8)
    pil_pred_color = Image.fromarray(pred_color_filtered)

    # draw bg box
    for target_label in target_labels_bg:
        box_wall = find_boxes(pred_filtered, target_label=target_label)
        visualize_box(pil_pred_color, box_wall, 'red')

    # draw base box
    for target_label in target_labels_base:
        box_floor = find_boxes(pred_filtered, target_label=target_label)
        visualize_box(pil_pred_color, box_floor, 'blue')

    pil_pred_color.save(
        os.path.join(cfg.TEST.result, f'{filename}_box.jpg')
    )

    # filter small objects
    labels = skimage.measure.label(pred_filtered, background=255, connectivity=1)
    erasered_labels = []
    uniques, counts = np.unique(labels, return_counts=True)
    for idx in np.argsort(counts)[::-1]:
        if (counts[idx] / pixs) >= 0.01:
            erasered_labels.append(uniques[idx])
    labels = filter_labels(labels, erasered_labels)
    idx = (labels == -1)
    pred_filtered[idx] = labels[idx]
    pred_color_without_small_objects = colorEncode(pred_filtered, colors).astype(np.uint8)
    pil_pred_color = Image.fromarray(pred_color_without_small_objects)

    bg_boxes = []
    for target_label in target_labels_bg:
        box = find_boxes(pred_filtered, target_label=target_label)
        visualize_box(pil_pred_color, box, 'red')
        bg_boxes.append(box)

    base_boxes = []
    for target_label in target_labels_base:
        box = find_boxes(pred_filtered, target_label=target_label)
        visualize_box(pil_pred_color, box, 'blue')
        base_boxes.append(box)

    pil_pred_color.save(
        os.path.join(cfg.TEST.result, f'{filename}_box_without_small.jpg')
    )

    bg_max_box = find_max_box(bg_boxes)
    base_max_box = find_max_box(base_boxes)

    if len(bg_max_box) > 0 and len(base_max_box) > 0:
        # iou = get_iou(new_bbox_wall, new_bbox_floor)
        intersection = get_intersection(bg_max_box, base_max_box)
    else:
        intersection = 0
    data = {
        'bg_max_box': bg_max_box,
        'base_max_box': base_max_box,
        'intersection': intersection
    }
    with open(os.path.join(cfg.TEST.result, f"{filename}.json"), 'w') as f_out:
        json.dump(data, f_out)


def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )

        pbar.update(1)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, gpu)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image path, or a directory name"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # generate testing image list
    if os.path.isdir(args.imgs):
        imgs = find_recursive(args.imgs, ext='')
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
