
import os
from easydict import EasyDict as edict

__C = edict()

cfg = __C
__C.RANDOM_SEED = 123
__C.DATA = edict()
__C.DATA.MOT15NAME = []
__C.DATA.VALNAME = ['MOT17-04', 'MOT17-11']
__C.DATA.VALFRAMENUM = [1050, 900]
__C.DATA.MOT17NAME = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
__C.DATA.MOT17FRAMENUM = [600, 1050, 837, 525, 654, 900, 750]
__C.DATA.MOT17ALLNAME = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
__C.DATA.MOT17ALLFRAMENUM = [600, 1050, 837, 525, 654, 900, 750]
__C.DATA.STATIC = []
__C.DATA.MOVING = []
__C.DATA.REID_DIR = ''
__C.DATA.MAXAGE = 100
__C.DATA.PATH_TO_DATA_DIR = 'data/MOT17/train/'
__C.DATA.PATH_TO_NPY_DIR = 'npy/npytrain_gt/'
__C.DATA.AUGMENTATION = False

__C.TRAIN = edict()
__C.TRAIN.MOVING_AVERAGE_ALPHA = 0.9
__C.TRAIN.MODE = 'onlytrain' # 'trainval' or 'all' or 'onlytrain'
__C.TRAIN.LR = 0.00001
# __C.TRAIN.LR = 0.00001
__C.TRAIN.WEIGHT_DECAY = 0.00001
__C.TRAIN.MAXEPOCH = 2
__C.TRAIN.BATCH_SIZE = 1
__C.STATISTIC_STEP = 50
__C.TRAIN.VOTING_ALPHA = 1000
__C.model_dir = "experiments/mot17static"
__C.save_checkpoint = True
__C.warmstart_path = "/root/deep_sort/experiments/1022-2/trainall/params/0001"
__C.resume = False
__C.TBDIR = 'tensorboard/'
#__C.gpu_id = 1,2,3
