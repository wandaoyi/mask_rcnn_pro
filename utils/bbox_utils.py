#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/01 00:13
# @Author   : WanDaoYi
# @FileName : bbox_utils.py
# ============================================

import numpy as np
import tensorflow as tf
from utils.image_utils import ImageUtils
from utils.mask_util import MaskUtil
from config import cfg


class BboxUtil(object):

    def __init__(self):
        self.image_utils = ImageUtils()
        self.mask_util = MaskUtil()
        pass

    # 提取 bounding boxes
    def extract_bboxes(self, mask):
        """
        :param mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        :return: bbox array [num_instances, (y1, x1, y2, x2)]
        """
        # 获取无类别的 instances 值，只区分前景和背景，类别在 目标检测的时候区分
        num_instance = mask.shape[-1]
        # 初始化 boxes
        boxes = np.zeros([num_instance, 4], dtype=np.int32)

        for i in range(num_instance):
            m = mask[:, :, i]

            # bounding box
            # x 轴方向
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            # y 轴方向
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                # 就是 x2 和 y2 不包含在 box 内，如 x1 = 1, x2 = 5, y1 = 1, y2 = 5
                # 围起来的面积右下角不包含 (5, 5)，所以加 1，以使 右下角超出 mask 面积外
                x2 += 1
                y2 += 1
                pass
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
                pass
            boxes[i] = np.array([y1, x1, y2, x2])
            pass
        return boxes.astype(np.int32)
        pass

    # 计算 box 的 IOU
    def compute_iou(self, box, boxes):
        """
        :param box: (y1, x1, y2, x2)
        :param boxes: [N, (y1, x1, y2, x2)]
        :return: iou
        """
        # 计算 box 面积
        # area = (x2 - x1) * (y2 - y1)
        box_area = (box[3] - box[1]) * (box[2] - box[0])
        boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # 计算交面积
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

        # 计算 IOU
        union = box_area + boxes_area[:] - intersection[:]
        # iou = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        iou = intersection / union
        return iou
        pass

    # 计算 boxes 的 IOU 重叠率
    def compute_overlaps(self, boxes1, boxes2):
        """
        :param boxes1: [N, (y1, x1, y2, x2)]
        :param boxes2: [N, (y1, x1, y2, x2)]
        :return:
        """
        # 定义覆盖率结构
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]
            overlaps[:, i] = self.compute_iou(box2, boxes1)
            pass
        return overlaps
        pass

    def overlaps_graph(self, boxes1, boxes2):
        """
            Computes IoU overlaps between two sets of boxes.
        :param boxes1: [N, (y1, x1, y2, x2)].
        :param boxes2: [N, (y1, x1, y2, x2)].
        :return:
        """
        # 1. Tile boxes2 and repeat boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.
        # TF doesn't have an equivalent to np.repeat() so simulate it
        # using tf.tile() and tf.reshape.
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute intersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps
        pass

    # 非极大值抑制
    def non_max_suppression(self, boxes, scores, threshold):
        """
        :param boxes: [N, (y1, x1, y2, x2)]. 注意，(y2, x2) 处于 box 之外
        :param scores: box 的得分
        :param threshold: IOU 阈值
        :return:
        """

        assert boxes.shape[0] > 0
        if boxes.dtype.kind != "f":
            boxes = boxes.astype(np.float32)
            pass

        # Get indices of boxes sorted by scores (highest first)
        ixs = scores.argsort()[::-1]

        pick = []
        while len(ixs) > 0:
            # Pick top box and add its index to the list
            i = ixs[0]
            pick.append(i)
            # Compute IoU of the picked box with the rest
            iou = self.compute_iou(boxes[i], boxes[ixs[1:]])
            # Identify boxes with IoU over the threshold. This
            # returns indices into ixs[1:], so add 1 to get
            # indices into ixs.
            remove_ixs = np.where(iou > threshold)[0] + 1
            # Remove indices of the picked and overlapped boxes.
            ixs = np.delete(ixs, remove_ixs)
            ixs = np.delete(ixs, 0)

        return np.array(pick, dtype=np.int32)
        pass

    # boxes 信息转换，bounding box regression
    # tx = (x − xa) / wa , ty = (y − ya) / ha,
    # tw = log(w / wa), th = log(h / ha)
    def apply_box_deltas(self, boxes, deltas):
        """
        :param boxes: [N, (y1, x1, y2, x2)]. 注意，(y2, x2) 处于 box 之外
        :param deltas: [N, (dy, dx, log(dh), log(dw))]
        :return:
        """
        boxes = boxes.astype(np.float32)
        # Convert to y, x, h, w
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        center_y = boxes[:, 0] + 0.5 * height
        center_x = boxes[:, 1] + 0.5 * width

        # Apply deltas
        center_y += deltas[:, 0] * height
        center_x += deltas[:, 1] * width
        height *= np.exp(deltas[:, 2])
        width *= np.exp(deltas[:, 3])

        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width

        return np.stack([y1, x1, y2, x2], axis=1)
        pass

    # boxes 与 ground truth 信息转换 tf 图，bounding box regression
    # 参考 bounding box regression 的公式
    def box_refinement_graph(self, box, gt_box):
        """
        :param box: [N, (y1, x1, y2, x2)]
        :param gt_box: [N, (y1, x1, y2, x2)]
        :return:
        """
        box = tf.cast(box, tf.float32)
        gt_box = tf.cast(gt_box, tf.float32)

        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        center_y = box[:, 0] + 0.5 * height
        center_x = box[:, 1] + 0.5 * width

        gt_height = gt_box[:, 2] - gt_box[:, 0]
        gt_width = gt_box[:, 3] - gt_box[:, 1]
        gt_center_y = gt_box[:, 0] + 0.5 * gt_height
        gt_center_x = gt_box[:, 1] + 0.5 * gt_width

        dy = (gt_center_y - center_y) / height
        dx = (gt_center_x - center_x) / width
        dh = tf.log(gt_height / height)
        dw = tf.log(gt_width / width)

        result = tf.stack([dy, dx, dh, dw], axis=1)
        return result
        pass

    # boxes 与 ground truth 信息转换，bounding box regression
    # 参考 bounding box regression 的公式
    def box_refinement(self, box, gt_box):
        """
        :param box: [N, (y1, x1, y2, x2)], 假设 (y2, x2) 处于 box 之外
        :param gt_box: [N, (y1, x1, y2, x2)]
        :return:
        """
        box = box.astype(np.float32)
        gt_box = gt_box.astype(np.float32)

        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        center_y = box[:, 0] + 0.5 * height
        center_x = box[:, 1] + 0.5 * width

        gt_height = gt_box[:, 2] - gt_box[:, 0]
        gt_width = gt_box[:, 3] - gt_box[:, 1]
        gt_center_y = gt_box[:, 0] + 0.5 * gt_height
        gt_center_x = gt_box[:, 1] + 0.5 * gt_width

        dy = (gt_center_y - center_y) / height
        dx = (gt_center_x - center_x) / width
        dh = np.log(gt_height / height)
        dw = np.log(gt_width / width)

        return np.stack([dy, dx, dh, dw], axis=1)
        pass

    # 将框从像素坐标转为标准坐标
    def norm_boxes_graph(self, boxes, shape):
        """
        :param boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
        :param shape: [..., (height, width)] in pixels
        :return: [..., (y1, x1, y2, x2)] in normalized coordinates
        注意：像素坐标 (y2，x2) 在框外。但在标准化坐标系下它在盒子里。
        """
        h, w = tf.split(tf.cast(shape, tf.float32), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
        shift = tf.constant([0., 0., 1., 1.])
        return tf.divide(boxes - shift, scale)
        pass

    def norm_boxes(self, boxes, shape):
        """
            Converts boxes from pixel coordinates to normalized coordinates.
        :param boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
        :param shape: [..., (height, width)] in pixels
        :return: [N, (y1, x1, y2, x2)] in normalized coordinates
            Note: In pixel coordinates (y2, x2) is outside the box.
                  But in normalized coordinates it's inside the box.
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.divide((boxes - shift), scale).astype(np.float32)
        pass

    def denorm_boxes(self, boxes, shape):
        """
            Converts boxes from normalized coordinates to pixel coordinates.
        :param boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
        :param shape: [..., (height, width)] in pixels
        :return: [N, (y1, x1, y2, x2)] in pixel coordinates

        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
             coordinates it's inside the box.
        """
        h, w = shape
        scale = np.array([h - 1, w - 1, h - 1, w - 1])
        shift = np.array([0, 0, 1, 1])
        return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)
        pass

    def apply_box_deltas_graph(self, boxes, deltas):
        """
            Applies the given deltas to the given boxes.
        :param boxes: [N, (y1, x1, y2, x2)] boxes to update
        :param deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        :return:
        """
        # Convert to y, x, h, w
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        center_y = boxes[:, 0] + 0.5 * height
        center_x = boxes[:, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, 0] * height
        center_x += deltas[:, 1] * width
        height *= tf.exp(deltas[:, 2])
        width *= tf.exp(deltas[:, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
        return result
        pass

    def clip_boxes_graph(self, boxes, window):
        """
        :param boxes: [N, (y1, x1, y2, x2)]
        :param window: [4] in the form y1, x1, y2, x2
        :return:
        """
        # Split
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
        clipped.set_shape((clipped.shape[0], 4))
        return clipped
        pass

    def load_image_gt(self, data, image_id, augmentation=None, use_mini_mask=False):
        """
            Load and return ground truth data for an image (image, mask, bounding boxes).
        :param data: The Dataset object to pick data from
        :param image_id: GT bounding boxes and masks for image id.
        :param augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
                            right/left 50% of the time.
        :param use_mini_mask: If False, returns full-size masks that are the same height
                            and width as the original image. These can be big, for example
                            1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
                            224x224 and are generated by extracting the bounding box of the
                            object and resizing it to MINI_MASK_SHAPE.
        :return:
            image: [height, width, 3]
            shape: the original shape of the image before resizing and cropping.
            class_ids: [instance_count] Integer class IDs
            bbox: [instance_count, (y1, x1, y2, x2)]
            mask: [height, width, instance_count]. The height and width are those
                of the image unless use_mini_mask is True, in which case they are
                defined in MINI_MASK_SHAPE.
        """

        # Load image and mask
        image_path = data.image_info_list[image_id]["path"]
        image = self.image_utils.load_image(image_path)
        mask, class_ids = self.mask_util.load_mask(data, image_id)

        original_shape = image.shape

        image, window, scale, padding, crop = self.image_utils.resize_image(image,
                                                                            min_dim=cfg.COMMON.IMAGE_MIN_DIM,
                                                                            min_scale=cfg.COMMON.IMAGE_MIN_SCALE,
                                                                            max_dim=cfg.COMMON.IMAGE_MAX_DIM,
                                                                            resize_mode=cfg.COMMON.IMAGE_RESIZE_MODE)
        mask = self.mask_util.resize_mask(mask, scale, padding, crop)

        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation:
            import imgaug

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in cfg.TRAIN.MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape

            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)

            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask = det.augment_image(mask.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))

            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

            # Change mask back to bool
            mask = mask.astype(np.bool)
            pass

        # Note that some boxes might be all zeros if the corresponding mask got cropped out.
        # and here is to filter them out
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        class_ids = class_ids[_idx]
        # Bounding boxes. Note that some boxes might be all zeros
        # if the corresponding mask got cropped out.
        # bbox: [num_instances, (y1, x1, y2, x2)]
        bbox = self.extract_bboxes(mask)

        # Active classes
        # Different datasets have different classes, so track the
        # classes supported in the dataset of this image.
        active_class_ids = np.zeros([data.class_num], dtype=np.int32)
        source_class_ids = data.source_class_ids[data.image_info_list[image_id]["source"]]
        active_class_ids[source_class_ids] = 1

        # Resize masks to smaller size to reduce memory usage
        if use_mini_mask:
            mask = self.mask_util.minimize_mask(bbox, mask, cfg.TRAIN.MINI_MASK_SHAPE)

        # Image meta data
        image_meta = self.image_utils.compose_image_meta(image_id, original_shape, image.shape,
                                                         window, scale, active_class_ids)

        return image, image_meta, class_ids, bbox, mask
        pass











