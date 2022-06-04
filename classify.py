import argparse
from curses import tparm
import fnmatch
import json
import os

import pandas as pd
from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-i", "--input_dir",
    help="input directory"
  )
  parser.add_argument(
    "-o", "--output_filepath",
    help="output filepath of csv",
    type=str,
    default="hrnetv2"
  )
  args = parser.parse_args()

  output_dict = []
  for root, _, files in os.walk(args.input_dir):
    files.sort()
    for f in tqdm(files):
      filename, ext = os.path.splitext(f)
      json_filepath = os.path.join('output', f"{filename}.json")
      data = []
      if not os.path.exists(json_filepath):
        continue
      with open(json_filepath) as f_in:
        data = json.load(f_in)

      bg_is_above_base = 0
      bg_max_box = data['bg_max_box']
      base_max_box = data['base_max_box']
      intersection = data['intersection']
      if len(bg_max_box) > 0 and len(base_max_box) > 0:
        if intersection == 0 and bg_max_box['y2'] < base_max_box['y1']:
            bg_is_above_base = 1

      # Classify view
      classification_type = ""
      if (intersection > 0 and intersection < 1) or (intersection == 0 and bg_is_above_base == 1):
        classification_type = "Product_Straight"
      else:
        classification_type = "Product_Top"

      DATA = {
        'filename': f,
        'intersection': intersection,
        'bg_is_above_base': bg_is_above_base,
        'classification_type': classification_type
      }
      output_dict.append(DATA)

  df = pd.DataFrame(output_dict)
  df.to_csv(args.output_filepath, index=False)

if __name__ == "__main__":
  main()
