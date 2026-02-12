"""
get token for every seq on the NuScenes dataset
"""

import os, json, sys
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

FIRST_TOKEN_ROOT_PATH = '../utils'

from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_gt
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.config import config_factory as track_configs

dataset_path='/data/wxx/nuscenes/v1.0-trainval/'

nusc = NuScenes(version="v1.0-trainval",verbose=True, dataroot=dataset_path)
cfg = track_configs("tracking_nips_2019")
gt_boxes = load_gt(nusc, "val", TrackingBox, verbose=True)

gt = gt_boxes.serialize()


# write token table
os.makedirs(FIRST_TOKEN_ROOT_PATH , exist_ok=True)
FIRST_TOKEN_PATH = FIRST_TOKEN_ROOT_PATH + "/gt_val.json"
print(f"write token table to {FIRST_TOKEN_PATH}")
json.dump(gt, open(FIRST_TOKEN_PATH, "w")) 
