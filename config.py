#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/04/15 22:53
# @Author   : WanDaoYi
# @FileName : config.py
# ============================================

import os
from easydict import EasyDict as edict

# mask_rcnn_coco.h5 预训练模型下载地址: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

__C = edict()
# Consumers can get config by: from config import cfg
cfg = __C

# common options 公共配置文件
__C.COMMON = edict()

# 论文中的模型 url
__C.COMMON.COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

# 相对路径 当前路径
__C.COMMON.RELATIVE_PATH = "./"

# mask 默认背景 类别, 背景为第一个类别
__C.COMMON.DEFAULT_CLASS_INFO = [{"source": "", "id": 0, "name": "BG"}]

__C.COMMON.DATA_SET_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset")

# 原始图像 文件 路径
__C.COMMON.IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/images")
# labelme 生成的 json 注释文件 路径
__C.COMMON.JSON_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/ann_json")

# 是否删除已有文件，True 为删除，False 为不删除
__C.COMMON.FILE_EXISTS_FLAG = True

# 数据划分比例
__C.COMMON.TEST_PERCENT = 0.7
__C.COMMON.VAL_PERCENT = 0.2
__C.COMMON.TEST_PERCENT = 0.1

# 数据来源
__C.COMMON.DATA_SOURCE = "our_data"

# 文件后缀名
__C.COMMON.JSON_SUFFIX = ".json"
__C.COMMON.PNG_SUFFIX = ".png"
__C.COMMON.JPG_SUFFIX = ".jpg"
__C.COMMON.TXT_SUFFIX = ".txt"

# 划分数据的保存路径
__C.COMMON.TRAIN_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/train_data.txt")
__C.COMMON.VAL_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/val_data.txt")
__C.COMMON.TEST_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/test_data.txt")

__C.COMMON.LOGS_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "logs")

# coco_class_names.txt 文件路径
__C.COMMON.COCO_CLASS_NAMES_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/coco_class_names.txt")
__C.COMMON.OUR_CLASS_NAMES_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/our_class_names.txt")

# Input image resizing
# Generally, use the "square" resizing mode for training and predicting
# and it should work well in most cases. In this mode, images are scaled
# up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
# scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
# padded with zeros to make it a square so multiple images can be put
# in one batch.
# Available resizing modes:
# none:   No resizing or padding. Return the image unchanged.
# square: Resize and pad with zeros to get a square image
#         of size [max_dim, max_dim].
# pad64:  Pads width and height with zeros to make them multiples of 64.
#         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
#         up before padding. IMAGE_MAX_DIM is ignored in this mode.
#         The multiple of 64 is needed to ensure smooth scaling of feature
#         maps up and down the 6 levels of the FPN pyramid (2**6=64).
# crop:   Picks random crops from the image. First, scales the image based
#         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
#         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
#         IMAGE_MAX_DIM is not used in this mode.
__C.COMMON.IMAGE_RESIZE_MODE = "square"
__C.COMMON.IMAGE_MIN_DIM = 800
__C.COMMON.IMAGE_MAX_DIM = 1024

# Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
# up scaling. For example, if set to 2 then images are scaled up to double
# the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
# However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
__C.COMMON.IMAGE_MIN_SCALE = 0

# 是否 crop 操作，True 为 crop
__C.COMMON.CROP_FLAG = False
# 训练输入图像的 shape
if __C.COMMON.CROP_FLAG:
    # [h, w, c]
    __C.COMMON.IMAGE_SHAPE = [800, 800, 3]
else:
    __C.COMMON.IMAGE_SHAPE = [1024, 1024, 3]

# 1 background + n classes
# __C.COMMON.CLASS_NUM = 1 + 80
__C.COMMON.CLASS_NUM = 1 + 1

# image_id(1维) + original_image_shape(3维) + image_shape(3维) + image_coor(y1, x1, y2, x2)(4维) +
# scale(1维) + class_num(类别数)
# 参考 compose_image_meta() 方法
__C.COMMON.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + __C.COMMON.CLASS_NUM

# backbone 支持 resNet50 和 resNet101
__C.COMMON.BACKBONE = "resNet101"

# The strides of each layer of the FPN Pyramid. These values
# are based on a resNet101 backbone.
__C.COMMON.BACKBONE_STRIDES = [4, 8, 16, 32, 64]

# Train or freeze batch normalization layers
#     None: Train BN layers. This is the normal mode
#     False: Freeze BN layers. Good when using a small batch size
#     True: (don't use). Set layer in training mode even when predicting
# Defaulting to False since batch size is often small
__C.COMMON.TRAIN_FLAG = False

# Size of the top-down layers used to build the feature pyramid
__C.COMMON.TOP_DOWN_PYRAMID_SIZE = 256

# Length of square anchor side in pixels
__C.COMMON.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

# Ratios of anchors at each cell (width/height)
# A value of 1 represents a square anchor, and 0.5 is a wide anchor
__C.COMMON.RPN_ANCHOR_RATIOS = [0.5, 1, 2]

# Anchor stride
# If 1 then anchors are created for each cell in the backbone feature map.
# If 2, then anchors are created for every other cell, and so on.
__C.COMMON.RPN_ANCHOR_STRIDE = 1

# Bounding box refinement standard deviation for RPN and final detections.
__C.COMMON.RPN_BBOX_STD_DEV = [0.1, 0.1, 0.2, 0.2]
__C.COMMON.BBOX_STD_DEV = [0.1, 0.1, 0.2, 0.2]

# Image mean (RGB)
__C.COMMON.MEAN_PIXEL = [123.7, 116.8, 103.9]

# ROIs kept after tf.nn.top_k and before non-maximum suppression
__C.COMMON.PRE_NMS_LIMIT = 6000

# Non-max suppression threshold to filter RPN proposals.
# You can increase this during training to generate more propsals.
__C.COMMON.RPN_NMS_THRESHOLD = 0.7

# Minimum probability value to accept a detected instance
# ROIs below this threshold are skipped
__C.COMMON.DETECTION_MIN_CONFIDENCE = 0.7

# Pooled ROIs
__C.COMMON.POOL_SIZE = 7

__C.COMMON.MASK_POOL_SIZE = 14

# Size of the fully-connected layers in the classification graph
__C.COMMON.FPN_CLASS_FC_LAYERS_SIZE = 1024

# NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
__C.COMMON.GPU_COUNT = 1

# Loss weights for more precise optimization.
# Can be used for R-CNN training setup.
__C.COMMON.LOSS_WEIGHTS = {"rpn_class_loss": 1.,
                           "rpn_bbox_loss": 1.,
                           "mrcnn_class_loss": 1.,
                           "mrcnn_bbox_loss": 1.,
                           "mrcnn_mask_loss": 1.
                           }

# mask train options 训练配置文件
__C.TRAIN = edict()

__C.TRAIN.DATA_SOURCE = "coco"

__C.TRAIN.DATA_SOURCE_INFO = "our_data"

__C.TRAIN.COCO_TRAIN_ANN_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/train_data.json")
__C.TRAIN.COCO_TRAIN_IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/images")

__C.TRAIN.COCO_VAL_ANN_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/val_data.json")
__C.TRAIN.COCO_VAL_IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/images")

__C.TRAIN.MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "models/mask_rcnn_coco.h5")
__C.TRAIN.SAVE_MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "models/mask_rcnn_coco_{epoch:04d}.h5")
__C.TRAIN.LOGS_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "logs")

# If enabled, resizes instance masks to a smaller size to reduce
# memory load. Recommended when using high-resolution images.
__C.TRAIN.USE_MINI_MASK = True
# (height, width) of the mini-mask
__C.TRAIN.MINI_MASK_SHAPE = (56, 56)

# Maximum number of ground truth instances to use in one image
__C.TRAIN.MAX_GT_INSTANCES = 100

# Shape of output mask
# To change this you also need to change the neural network mask branch
__C.TRAIN.MASK_SHAPE = [28, 28]

# train batch size
__C.TRAIN.BATCH_SIZE = 1

# learning rate for rough train
__C.TRAIN.ROUGH_LEARNING_RATE = 0.001
# learning rate for fine-tuning
__C.TRAIN.FINE_LEARNING_RATE = 0.0001

__C.TRAIN.WEIGHT_DECAY = 0.0001

# momentum 超参
__C.TRAIN.LEARNING_MOMENTUM = 0.9

# Gradient norm clipping
__C.TRAIN.GRADIENT_CLIP_NORM = 5.0

# n_epoch for rough train
__C.TRAIN.FIRST_STAGE_N_EPOCH = 32
# n_epoch for convergence loss
__C.TRAIN.MIDDLE_STAGE_N_EPOCH = 128
# n_epoch for fine-tuning
__C.TRAIN.LAST_STAGE_N_EPOCH = 256

# Training network heads
__C.TRAIN.HEADS_LAYERS = "heads"
# Fine tune ResNet stage 4 and up
__C.TRAIN.FOUR_MORE_LAYERS = "4+"
# Fine tune all layers
__C.TRAIN.ALL_LAYERS = "all"

# Number of training steps per epoch
# This doesn't need to match the size of the training set. Tensorboard
# updates are saved at the end of each epoch, so setting this to a
# smaller number means getting more frequent TensorBoard updates.
# Validation stats are also calculated at each epoch end and they
# might take a while, so don't set this too small to avoid spending
# a lot of time on validation stats.
__C.TRAIN.STEPS_PER_EPOCH = 1000

# Number of validation steps to run at the end of every training epoch.
# A bigger number improves accuracy of validation stats, but slows
# down the training.
__C.TRAIN.VALIDATION_STEPS = 50

# ROIs kept after non-maximum suppression
__C.TRAIN.POST_NMS_ROIS = 2000

# Number of ROIs per image to feed to classifier/mask heads
# The Mask RCNN paper uses 512 but often the RPN doesn't generate
# enough positive proposals to fill this and keep a positive:negative
# ratio of 1:3. You can increase the number of proposals by adjusting
# the RPN NMS threshold.
__C.TRAIN.ROIS_PER_IMAGE = 200

# Percent of positive ROIs used to train classifier/mask heads
__C.TRAIN.ROI_POSITIVE_RATIO = 0.33

# Use RPN ROIs or externally generated ROIs for training
# Keep this True for most situations. Set to False if you want to train
# the head branches on ROI generated by code rather than the ROIs from
# the RPN. For example, to debug the classifier head without having to
# train the RPN.
__C.TRAIN.USE_RPN_ROIS = True

# Pre-defined layer regular expressions
#                         heads: all layers but the backbone
__C.TRAIN.LAYER_REGEX = {"heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
                         # 3+, 4+, 5+: From a specific ResNet stage and up
                         "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
                         "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
                         "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
                         # all: All layers
                         "all": ".*",
                         }

# Augmenters that are safe to apply to masks
# Some, such as Affine, have settings that make them unsafe, so always
# test your augmentation on masks
__C.TRAIN.MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                             "Fliplr", "Flipud", "CropAndPad",
                             "Affine", "PiecewiseAffine"]

# How many anchors per image to use for RPN training
__C.TRAIN.ANCHORS_PER_IMAGE = 256

# mask_test options 测试配置文件
__C.TEST = edict()

# model 文件路径
__C.TEST.COCO_MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "models/mask_rcnn_coco_0001.h5")
__C.TEST.SAVE_MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "models")
__C.TEST.TEST_INFO_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "infos/test.txt")

__C.TEST.TEST_IMAGE_FILE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/test_image")
__C.TEST.OUTPUT_IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "output_info/images")

__C.TEST.POST_NMS_ROIS = 1000

# TEST batch size
__C.TEST.BATCH_SIZE = 1

# Max number of final detections
__C.TEST.DETECTION_MAX_INSTANCES = 100

__C.TEST.DETECTION_NMS_THRESHOLD = 0.3
