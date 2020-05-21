#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/12 15:40
# @Author   : WanDaoYi
# @FileName : backbone.py
# ============================================

import keras.layers as kl
from m_rcnn import common
from config import cfg


def resnet_graph(input_image, architecture, stage5=False):
    """
        resNet 背骨图，没什么好说的，数好参数就好了。
    :param input_image: input image info
    :param architecture: Can be resNet50 or resNet101
    :param stage5: Boolean. If False, stage5 of the network is not created
    :return: [c1, c2, c3, c4, c5]
    """
    train_flag = cfg.COMMON.TRAIN_FLAG
    assert architecture in ["resNet50", "resNet101"]
    # Stage 1
    x = kl.ZeroPadding2D((3, 3))(input_image)
    x = kl.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = common.BatchNorm(name='bn_conv1')(x, training=train_flag)
    x = kl.Activation('relu')(x)
    c1 = x = kl.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = common.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_flag=train_flag)
    x = common.identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_flag=train_flag)
    c2 = x = common.identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_flag=train_flag)

    # Stage 3
    x = common.conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_flag=train_flag)
    x = common.identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_flag=train_flag)
    x = common.identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_flag=train_flag)
    c3 = x = common.identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_flag=train_flag)

    # Stage 4
    x = common.conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_flag=train_flag)
    block_count = {"resNet50": 5, "resNet101": 22}[architecture]
    for i in range(block_count):
        x = common.identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_flag=train_flag)
    c4 = x

    # Stage 5
    if stage5:
        x = common.conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_flag=train_flag)
        x = common.identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_flag=train_flag)
        c5 = common.identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_flag=train_flag)
    else:
        c5 = None

    return [c1, c2, c3, c4, c5]

