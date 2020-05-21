#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/13 12:08
# @Author   : WanDaoYi
# @FileName : anchor_utils.py
# ============================================

import numpy as np
from utils.misc_utils import MiscUtils
from utils.bbox_utils import BboxUtil
from config import cfg


class AnchorUtils(object):

    def __init__(self):
        self.misc_utils = MiscUtils()
        self.bbox_utils = BboxUtil()

        # Cache anchors and reuse if image shape is the same
        self._anchor_cache = {}
        # self.anchors = None
        pass

    def get_anchors(self, image_shape):
        """
        :return: Returns anchor pyramid for the given image size
        """
        if tuple(image_shape) not in self._anchor_cache:
            # Generate Anchors
            anchor = self.generate_pyramid_anchors(image_shape)

            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            # self.anchors = anchor

            self._anchor_cache[tuple(image_shape)] = self.bbox_utils.norm_boxes(anchor, image_shape[:2])
            pass

        return self._anchor_cache[tuple(image_shape)]
        pass

    def generate_pyramid_anchors(self, image_shape):
        """
            Generate anchors at different levels of a feature pyramid.
            Each scale is associated with a level of the pyramid,
            but each ratio is used in all levels of the pyramid.
        :param image_shape: [h, w, c]
        :return: anchors: [N, (y1, x1, y2, x2)]
            All generated anchors in one array.
            Sorted with the same order of the given scales.
            So, anchors of scale[0] come first, then anchors of scale[1], and so on.
        """

        backbone_strides = cfg.COMMON.BACKBONE_STRIDES
        # [N, (height, width)]. Where N is the number of stages
        backbone_shape = self.misc_utils.compute_backbone_shapes(image_shape, backbone_strides)

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        anchors = []
        scales = cfg.COMMON.RPN_ANCHOR_SCALES
        scales_len = len(scales)

        for i in range(scales_len):
            anchor_box = self.generate_anchors(scales[i], backbone_shape[i], backbone_strides[i])
            anchors.append(anchor_box)
            pass

        return np.concatenate(anchors, axis=0)
        pass

    # generate anchor box
    def generate_anchors(self, scales, backbone_shape, backbone_strides):
        """
        :param scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        :param backbone_shape: [height, width] spatial shape of the feature map over which to generate anchors.
        :param backbone_strides: Stride of the feature map relative to the image in pixels.
        :return: anchor box: Convert to corner coordinates (y1, x1, y2, x2)
        """
        # 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        ratios = cfg.COMMON.RPN_ANCHOR_RATIOS

        # Stride of anchors on the feature map. For example,
        # if the value is 2 then generate anchors for every other feature map pixel.
        anchor_stride = cfg.COMMON.RPN_ANCHOR_STRIDE

        # Get all combinations of scales and ratios
        scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
        scales = scales.flatten()
        ratios = ratios.flatten()

        # Enumerate heights and widths from scales and ratios
        heights = scales / np.sqrt(ratios)
        widths = scales * np.sqrt(ratios)

        # Enumerate shifts in feature space
        shifts_y = np.arange(0, backbone_shape[0], anchor_stride) * backbone_strides
        shifts_x = np.arange(0, backbone_shape[1], anchor_stride) * backbone_strides
        shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
        box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)
        return boxes
        pass







