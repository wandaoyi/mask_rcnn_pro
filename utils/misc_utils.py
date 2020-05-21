#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/13 12:06
# @Author   : WanDaoYi
# @FileName : misc_utils.py
# ============================================

import math
import numpy as np
import tensorflow as tf
from utils.bbox_utils import BboxUtil
from config import cfg


class MiscUtils(object):

    def __init__(self):
        self.bbox_util = BboxUtil()
        pass

    def compute_backbone_shapes(self, image_shape, backbone_strides):
        """
            Computes the width and height of each stage of the backbone network
        :param image_shape: [h, w, c]
        :param backbone_strides: The strides of each layer of the FPN Pyramid.
                                These values are based on a resNet101 backbone.
        :return: [N, (height, width)]. Where N is the number of stages
        """
        return np.array(
            [[int(math.ceil(image_shape[0] / stride)),
              int(math.ceil(image_shape[1] / stride))] for stride in backbone_strides])
        pass

    def batch_slice(self, inputs, graph_fn, batch_size, names=None):
        """
            Splits inputs into slices and feeds each slice to a copy of the given
            computation graph and then combines the results. It allows you to run a
            graph on a batch of inputs even if the graph is written to support one
            instance only.
        :param inputs: list of tensors. All must have the same first dimension length
        :param graph_fn: A function that returns a TF tensor that's part of a graph.
        :param batch_size: number of slices to divide the data into.
        :param names: If provided, assigns names to the resulting tensors.
        :return:
        """

        if not isinstance(inputs, list):
            inputs = [inputs]

        outputs = []
        for i in range(batch_size):
            inputs_slice = [x[i] for x in inputs]
            output_slice = graph_fn(*inputs_slice)
            if not isinstance(output_slice, (tuple, list)):
                output_slice = [output_slice]
            outputs.append(output_slice)
        # Change outputs from a list of slices where each is
        # a list of outputs to a list of outputs and each has
        # a list of slices
        outputs = list(zip(*outputs))

        if names is None:
            names = [None] * len(outputs)

        result = [tf.stack(o, axis=0, name=n)
                  for o, n in zip(outputs, names)]
        if len(result) == 1:
            result = result[0]

        return result
        pass

    def trim_zeros_graph(self, boxes, name='trim_zeros'):
        """
            Often boxes are represented with matrices of shape [N, 4] and
            are padded with zeros. This removes zero boxes.
        :param boxes: [N, 4] matrix of boxes.
        :param name:
        :return: non_zeros: [N] a 1D boolean mask identifying the rows to keep
        """

        non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
        boxes = tf.boolean_mask(boxes, non_zeros, name=name)
        return boxes, non_zeros
        pass

    def detection_targets_graph(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """
            Generates detection targets for one image. Subsamples proposals and
            generates target class IDs, bounding box deltas, and masks for each.
        :param proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates.
                          Might be zero padded if there are not enough proposals.
        :param gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        :param gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
        :param gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
        :return: Target ROIs and corresponding class IDs, bounding box shifts, and masks.
            rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
            class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
            deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
            masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
                   boundaries and resized to neural network output size.

            Note: Returned arrays might be zero padded if not enough target ROIs.
        """

        # Assertions
        asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"), ]

        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)
            pass

        # Remove zero padding
        proposals, _ = self.trim_zeros_graph(proposals, name="trim_proposals")
        gt_boxes, non_zeros = self.trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2, name="trim_gt_masks")

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = self.bbox_util.overlaps_graph(proposals, gt_boxes)

        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        crowd_overlaps = self.bbox_util.overlaps_graph(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(cfg.TRAIN.ROIS_PER_IMAGE * cfg.TRAIN.ROI_POSITIVE_RATIO)
        positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / cfg.TRAIN.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # Compute bbox refinement for positive ROIs
        deltas = self.bbox_util.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= np.array(cfg.COMMON.BBOX_STD_DEV)

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        # Compute mask targets
        boxes = positive_rois
        if cfg.TRAIN.USE_MINI_MASK:
            # Transform ROI coordinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32),
                                         boxes, box_ids,
                                         cfg.TRAIN.MASK_SHAPE)
        # Remove the extra dimension from masks.
        masks = tf.squeeze(masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = tf.round(masks)

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(cfg.TRAIN.ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        # roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

        return rois, roi_gt_class_ids, deltas, masks
        pass







