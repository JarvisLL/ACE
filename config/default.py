from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
from yacs.config import CfgNode as CN

_C = CN()

_C.DESCRIPTION = 'deeplabv3plus_citys'
# environment
_C.NO_CUDA = False
_C.GPUS = [0,]
_C.LOCAL_RANK = 0
_C.INIT_METHOD = 'env://'
# 

# cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = True

# common params for NETWORKS
_C.MODEL = CN()
_C.MODEL.NAME = 'deeplabv3plus'

# DATASET related params
_C.DATASET = CN()
_C.DATASET.NAME = 'mapillary'
_C.DATASET.IMG_TRANSFORM = False
_C.DATASET.BASE_SIZE = 1024
_C.DATASET.CROP_SIZE = 512
_C.DATASET.WORKERS = 4
_C.DATASET.PIN_MEMORY = True
_C.DATASET.IGNORE_INDEX = -1

# training
_C.TRAIN = CN()
_C.TRAIN.MIXED_PRECISION = False
_C.TRAIN.MIXED_OPT_LEVEL = "O1"
_C.TRAIN.TRAIN_SPLIT = 'train'
_C.TRAIN.DROP_RATE = 0.2
_C.TRAIN.SEG_LOSS = 'focalloss2d'
_C.TRAIN.EPOCHS = 40
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.LR = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.WARMUP_ITERS = 2000
_C.TRAIN.WARMUP_FACTOR = 1.0 / 3
_C.TRAIN.EVAL_EPOCHS = 5
_C.TRAIN.SKIP_EVAL = False
_C.TRAIN.IMG_TRANSFORM = False
_C.TRAIN.DTYPE = 'float32'
_C.TRAIN.LOG_STEP = 10
_C.TRAIN.SAVE_EPOCH = 5
_C.TRAIN.SAVE_DIR = './checkpoint/basemodel'
_C.TRAIN.RESUME = ''

# testing
_C.TEST = CN()
_C.TEST.MODE = 'testval' #'test' 'train' 'val' or 'testval'
_C.TEST.TEST_SPLIT = 'val'  #'train' 'val' 'trainval'
_C.TEST.MODEL_NAME = 'lpnetdepthwiseres26'
_C.TEST.TEST_BATCH_SIZE = 1
_C.TEST.PRETRAINED = ''