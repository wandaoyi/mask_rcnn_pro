#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/01 00:22
# @Author   : WanDaoYi
# @FileName : mask_util.py
# ============================================

import warnings
import numpy as np
import scipy.ndimage
from utils.image_utils import ImageUtils
from pycocotools import mask as coco_mask_utils
from config import cfg


class MaskUtil(object):

    def __init__(self):
        self.coco_model_url = cfg.COMMON.COCO_MODEL_URL
        self.image_utils = ImageUtils()
        pass

    # 计算两个 masks 的 IOU 重叠率
    def compute_overlaps_masks(self, masks1, masks2):
        """
        :param masks1: [Height, Width, instances]
        :param masks2: [Height, Width, instances]
        :return: 两个 masks 的 IOU 重叠率
        """
        # 如果其中一个 masks 为空，则返回 空 结果
        mask_flag = masks1.shape[-1] == 0 or masks2.shape[-1] == 0
        if mask_flag:
            return np.zeros((masks1.shape[-1], masks2.shape[-1]))
            pass

        # 将 masks 扁平化后并计算它们的面积
        masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
        area1 = np.sum(masks1, axis=0)
        area2 = np.sum(masks2, axis=0)

        # intersections and union
        intersections = np.dot(masks1.T, masks2)
        union = area1[:, None] + area2[None, :] - intersections
        overlaps = intersections / union

        return overlaps
        pass

    def annotation_2_mask(self, annotation, height, width):
        """
            Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :param annotation: annotation info
        :param height: image info of height
        :param width: image info of width
        :return: binary mask (numpy 2D array)
        """
        segment = annotation['segmentation']
        if isinstance(segment, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = coco_mask_utils.frPyObjects(segment, height, width)
            rle = coco_mask_utils.merge(rles)
            pass
        elif isinstance(segment['counts'], list):
            # uncompressed RLE
            rle = coco_mask_utils.frPyObjects(segment, height, width)
            pass
        else:
            # rle
            rle = segment['segmentation']
            pass
        mask = coco_mask_utils.decode(rle)
        return mask
        pass

    def load_mask(self, data, image_id):
        """
            Load instance masks for the given image.
            Different datasets use different ways to store masks. This
            function converts the different mask format to one format
            in the form of a bitmap [height, width, instances].
        :param data: The Dataset object to pick data from
        :param image_id: image id of image
        :return:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = data.image_info_list[image_id]

        instance_masks = []
        class_ids = []
        annotations = data.image_info_list[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:

            class_id = data.class_from_source_map["coco.{}".format(annotation['category_id'])]

            if class_id:
                m = self.annotation_2_mask(annotation, image_info["height"], image_info["width"])

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                    pass

                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

            pass

        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids
        pass

    def resize_mask(self, mask, scale, padding, crop=None):
        """
            resize a mask using the given scale and padding.
            Typically, you get the scale and padding from resize_image() to
            ensure both, the image and the mask, are resized consistently.
        :param mask:
        :param scale: mask scaling factor
        :param padding: Padding to add to the mask in the form
                        [(top, bottom), (left, right), (0, 0)]
        :param crop:
        :return:
        """
        # Suppress warning from scipy 0.13.0, the output shape of zoom() is
        # calculated with round() instead of int()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        if crop is not None:
            y, x, h, w = crop
            mask = mask[y:y + h, x:x + w]
        else:
            mask = np.pad(mask, padding, mode='constant', constant_values=0)
        return mask
        pass

    def minimize_mask(self, bbox, mask, mini_shape):
        """
            Resize masks to a smaller version to reduce memory load.
            Mini-masks can be resized back to image scale using expand_masks()
        :param bbox:
        :param mask:
        :param mini_shape:
        :return:
        """
        # 避免 传参 过来 是 list，在 cfg.TRAIN.MINI_MASK_SHAPE 获得的是 list
        mini_shape = tuple(mini_shape)
        mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
        for i in range(mask.shape[-1]):
            # Pick slice and cast to bool in case load_mask() returned wrong dtype
            m = mask[:, :, i].astype(bool)
            y1, x1, y2, x2 = bbox[i][:4]
            m = m[y1:y2, x1:x2]
            if m.size == 0:
                raise Exception("Invalid bounding box with area of zero")
            # Resize with bilinear interpolation
            m = self.image_utils.resize(m, mini_shape)
            mini_mask[:, :, i] = np.around(m).astype(np.bool)
        return mini_mask
        pass

    def unmold_mask(self, mask, bbox, image_shape):
        """
            Converts a mask generated by the neural network to a format similar
            to its original shape.
        :param mask: [height, width] of type float. A small, typically 28x28 mask.
        :param bbox: [y1, x1, y2, x2]. The box to fit the mask in.
        :param image_shape:
        :return: return a binary mask with the same size as the original image.
        """
        threshold = 0.5
        y1, x1, y2, x2 = bbox
        mask = self.image_utils.resize(mask, (y2 - y1, x2 - x1))
        mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

        # Put the mask in the right location.
        full_mask = np.zeros(image_shape[:2], dtype=np.bool)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask
        pass




