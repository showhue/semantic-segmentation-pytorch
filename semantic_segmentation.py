# System libs
import io
import os
import tempfile
import zipfile
from typing import List

# Numerical libs
import numpy as np
import requests
import skimage
import torch
import torch.nn as nn
from google.cloud import storage

from mit_semseg.config import cfg
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.lib.nn import user_scattered_collate
from mit_semseg.lib.utils import as_numpy
from mit_semseg.models import ModelBuilder, SegmentationModule


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

def classify(pred) -> str:
  labels = []

  # print predictions in descending order
  pred = np.int32(pred)
  pixs = pred.size
  uniques, counts = np.unique(pred, return_counts=True)
  for idx in np.argsort(counts)[::-1]:
    ratio = counts[idx] / pixs * 100
    if ratio > 0.1:
      labels.append(uniques[idx])

  # [0: wall, 2: sky, 3: floor, 6: road, 13: ground, 15:table]
  target_labels_bg = [0, 2]
  target_labels_base = [3, 6, 13, 15]
  pred_filtered = filter_labels(pred, labels=target_labels_bg+target_labels_base)

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

  bg_boxes = []
  for target_label in target_labels_bg:
    box = find_boxes(pred_filtered, target_label=target_label)
    bg_boxes.append(box)

  base_boxes = []
  for target_label in target_labels_base:
    box = find_boxes(pred_filtered, target_label=target_label)
    base_boxes.append(box)

  bg_max_box = find_max_box(bg_boxes)
  base_max_box = find_max_box(base_boxes)

  if len(bg_max_box) > 0 and len(base_max_box) > 0:
    intersection = get_intersection(bg_max_box, base_max_box)
  else:
    intersection = 0

  bg_is_above_base = False
  if len(bg_max_box) > 0 and len(base_max_box) > 0:
    if intersection == 0 and bg_max_box['y2'] < base_max_box['y1']:
      bg_is_above_base = True

  if (intersection > 0 and intersection < 1) or (intersection == 0 and bg_is_above_base == 1):
    return "Product_Straight"
  else:
    return "Product_Top"

class SemanticSegmentation():
  def __init__(self, cfg_file='./config/ade20k-hrnetv2.yaml') -> None:
    self._source_path = os.path.dirname(os.path.realpath(__file__))
    self._model_path = os.path.join(self._source_path, 'ckpt')
    if not os.path.exists(os.path.join(self._model_path, 'ade20k-hrnetv2-c1')):
      self._model_link = 'https://storage.googleapis.com/useeapi/models/semantic-segmentation-pytorch/ade20k-hrnetv2-c1.zip'
      r = requests.get(self._model_link)
      temp_weight_file = os.path.join(tempfile.gettempdir(), 'weight_temp_file.zip')
      with open(temp_weight_file, 'wb') as f:
        f.write(r.content)
      with zipfile.ZipFile(temp_weight_file, 'r') as zip_ref:
        zip_ref.extractall(self._model_path)
      os.remove(temp_weight_file)

    cfg.merge_from_file(cfg_file)

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    cfg.MODEL.weights_encoder = os.path.join(
      cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
      cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

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

    self.segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

  def forward(self, imgs: List[bytes]) -> str:
    cfg.list_test = [{'fpath_img': io.BytesIO(x)} for x in imgs]
    dataset_test = TestDataset(
      cfg.list_test,
      cfg.DATASET
    )
    loader_test = torch.utils.data.DataLoader(
      dataset_test,
      batch_size=cfg.TEST.batch_size,
      shuffle=False,
      collate_fn=user_scattered_collate,
      num_workers=0,
      drop_last=True
    )

    self.segmentation_module.eval()

    batch_data = list(loader_test)[0]
    # process data
    batch_data = batch_data[0]
    segSize = (batch_data['img_ori'].shape[0],
                batch_data['img_ori'].shape[1])
    img_resized_list = batch_data['img_data']

    with torch.no_grad():
      scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])

      for img in img_resized_list:
        feed_dict = batch_data.copy()
        feed_dict['img_data'] = img
        del feed_dict['img_ori']
        del feed_dict['info']

        # forward pass
        pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
        scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

      _, pred = torch.max(scores, dim=1)
      pred = as_numpy(pred.squeeze(0).cpu())

    return classify(pred)

def main():
  storage_client = storage.Client(project='ire-lab-200005')
  client = storage_client.get_bucket('recommendation-beta')

  ref_image_id = '1613992441_35cd0deb-f670-4531-a526-6e3b26efdbdc'
  key = os.path.join('ref-images', ref_image_id)
  blob = client.blob(key)

  import time
  t = time.time()
  model = SemanticSegmentation()
  classification_type = model.forward([blob.download_as_bytes()])
  print(classification_type)
  print(time.time() - t)

if __name__ == "__main__":
  main()
