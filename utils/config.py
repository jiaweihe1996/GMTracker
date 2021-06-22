"""Graph matching config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
from easydict import EasyDict as edict
import numpy as np

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# random seed used for data loading
__C.RANDOM_SEED = 123

__C.DATA = edict()
__C.DATA.MOT15NAME = ['ETH-Bahnhof', 'ETH-Sunnyday', 'KITTI-17', 'PETS09-S2L1', 'TUD-Stadtmitte']
__C.DATA.MOT15FRAMENUM = [500, 177, 72, 397, 89]

## fold 0
__C.DATA.FOLD0_TRAIN = ['MOT17-04', 'MOT17-05', 'MOT17-09','MOT17-11']
__C.DATA.FOLD0_TRAIN_NUM = [1050, 837, 525, 900]
__C.DATA.FOLD0_VAL = ['MOT17-02', 'MOT17-10', 'MOT17-13']
## fold 1
__C.DATA.FOLD1_TRAIN = ['MOT17-02', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-13']
__C.DATA.FOLD1_TRAIN_NUM = [600, 837, 525, 654, 750]
__C.DATA.FOLD1_VAL = ['MOT17-04', 'MOT17-11']
## fold 2
__C.DATA.FOLD2_TRAIN = ['MOT17-02', 'MOT17-04', 'MOT17-10', 'MOT17-11', 'MOT17-13']
__C.DATA.FOLD2_TRAIN_NUM = [600, 654, 900, 750]
__C.DATA.FOLD2_VAL = ['MOT17-05', 'MOT17-09']

__C.DATA.MOT17ALLNAME = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
__C.DATA.MOT17ALLFRAMENUM = [600, 1050, 837, 525, 654, 900, 750]
__C.DATA.STATIC = ['MOT17-01', 'MOT17-03', 'MOT17-08', 'MOT17-02', 'MOT17-04', 'MOT17-09']
__C.DATA.MOVING = ['MOT17-06', 'MOT17-07', 'MOT17-12', 'MOT17-14', 'MOT17-05', 'MOT17-10', 'MOT17-11', 'MOT17-13']
__C.DATA.REID_DIR = ''
__C.DATA.MAXAGE = 100
__C.DATA.PATH_TO_DATA_DIR = 'data/MOT17/train/'
__C.DATA.VAL_DIR = 'data/MOT17/train/'
__C.DATA.TEST_DIR = 'data/MOT17/test/'
__C.DATA.PATH_TO_NPY_DIR = 'npy/npytrain/'
__C.DATA.AUGMENTATION = False

__C.TRAIN = edict()
__C.TRAIN.MOVING_AVERAGE_ALPHA = 0.9
__C.TRAIN.MODE = 'onlytrain' # 'trainval' or 'all' or 'onlytrain'
__C.TRAIN.LR = 0.00008
__C.TRAIN.Optim = 'Adam'
__C.TRAIN.WEIGHT_DECAY = 0.00001
__C.TRAIN.MAXEPOCH = 4
__C.TRAIN.BATCH_SIZE = 1
__C.STATISTIC_STEP = 20
__C.TRAIN.VOTING_ALPHA = 1000
__C.save_model_dir = ""
__C.save_checkpoint = True
__C.warmstart_path = ""
__C.resume = False
__C.detections = ['DPM', 'FRCNN', 'SDP']
__C.RESULTS_DIR = 'result/final/'
__C.RESULTS_VAL_DIR = 'result/val/'
__C.moving_checkpoint_dir = 'experiments/moving/params/0001/'
__C.static_checkpoint_dir = 'experiments/static/params/0001/'