#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/12 15:42
# @Author   : WanDaoYi
# @FileName : mask_rcnn.py
# ============================================

import h5py
import numpy as np
import tensorflow as tf
import keras.backend as k
import keras.layers as kl
import keras.models as km
from utils.bbox_utils import BboxUtil
from utils.anchor_utils import AnchorUtils
from utils.image_utils import ImageUtils
from utils.mask_util import MaskUtil
from m_rcnn import common
from m_rcnn import backbone
from config import cfg

# Conditional import to support versions of Keras before 2.2
try:
    from keras.engine import saving
except ImportError:
    # Keras before 2.2 used the 'topology' namespace.
    from keras.engine import topology as saving


class MaskRCNN(object):

    def __init__(self, train_flag=True):
        """
        :param train_flag: 是否为训练，训练为 True，测试为 False
        """
        self.train_flag = train_flag
        self.bbox_util = BboxUtil()
        self.anchor_utils = AnchorUtils()
        self.image_utils = ImageUtils()
        self.mask_util = MaskUtil()

        # 模型 路径
        self.model_path = cfg.TRAIN.MODEL_PATH if self.train_flag else cfg.TEST.COCO_MODEL_PATH
        # batch size
        self.batch_size = cfg.TRAIN.BATCH_SIZE if self.train_flag else cfg.TEST.BATCH_SIZE
        # 模型保存路径
        self.save_model_path = cfg.TRAIN.SAVE_MODEL_PATH

        self.backbone = cfg.COMMON.BACKBONE
        self.backbone_strides = cfg.COMMON.BACKBONE_STRIDES
        # 输入图像
        self.image_shape = np.array(cfg.COMMON.IMAGE_SHAPE)
        # 用于构建特征金字塔的自顶向下层的大小
        self.top_down_pyramid_size = cfg.COMMON.TOP_DOWN_PYRAMID_SIZE

        self.rpn_anchor_stride = cfg.COMMON.RPN_ANCHOR_STRIDE
        self.rpn_anchor_ratios = cfg.COMMON.RPN_ANCHOR_RATIOS
        self.rpn_nms_threshold = cfg.COMMON.RPN_NMS_THRESHOLD

        self.class_num = cfg.COMMON.CLASS_NUM

        self.rois_per_image = cfg.TRAIN.ROIS_PER_IMAGE
        self.roi_positive_ratio = cfg.TRAIN.ROI_POSITIVE_RATIO

        self.keras_model = self.build()

        pass

    def build(self):

        # image shape
        h, w, c = self.image_shape[:]
        print("image_shape: {}".format(self.image_shape))

        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

            # Inputs
        input_image = kl.Input(shape=[None, None, c], name="input_image")
        input_image_meta = kl.Input(shape=[cfg.COMMON.IMAGE_META_SIZE], name="input_image_meta")

        # 训练
        if self.train_flag:

            # RPN GT
            input_rpn_match = kl.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = kl.Input(shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = kl.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)

            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = kl.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)

            # Normalize coordinates
            gt_boxes = kl.Lambda(lambda x: self.bbox_util.norm_boxes_graph(x, k.shape(input_image)[1:3]))(
                input_gt_boxes)

            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if cfg.TRAIN.USE_MINI_MASK:
                min_h, min_w = cfg.TRAIN.MINI_MASK_SHAPE[:]
                input_gt_masks = kl.Input(shape=[min_h, min_w, None], name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = kl.Input(shape=[h, w, None], name="input_gt_masks", dtype=bool)
                pass

            # anchor
            anchors = self.anchor_utils.get_anchors(self.image_shape)

            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (self.batch_size,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = kl.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)

            anchors = kl.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
            pass

        else:
            # Anchors in normalized coordinates
            anchors = kl.Input(shape=[None, 4], name="input_anchors")

            # 上面训练用到的参数，测试不需要，但是在 if else 里面定义一下，免得 undefined
            input_rpn_match = None
            input_rpn_bbox = None
            input_gt_class_ids = None
            gt_boxes = None
            input_gt_boxes = None
            input_gt_masks = None
            pass

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, c2, c3, c4, c5 = backbone.resnet_graph(input_image, self.backbone, stage5=True)

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        p5 = kl.Conv2D(self.top_down_pyramid_size, (1, 1), name='fpn_c5p5')(c5)
        p4 = kl.Add(name="fpn_p4add")([kl.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(p5),
                                       kl.Conv2D(self.top_down_pyramid_size, (1, 1), name='fpn_c4p4')(c4)])
        p3 = kl.Add(name="fpn_p3add")([kl.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(p4),
                                       kl.Conv2D(self.top_down_pyramid_size, (1, 1), name='fpn_c3p3')(c3)])
        p2 = kl.Add(name="fpn_p2add")([kl.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(p3),
                                       kl.Conv2D(self.top_down_pyramid_size, (1, 1), name='fpn_c2p2')(c2)])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        p2 = kl.Conv2D(self.top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p2")(p2)
        p3 = kl.Conv2D(self.top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p3")(p3)
        p4 = kl.Conv2D(self.top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p4")(p4)
        p5 = kl.Conv2D(self.top_down_pyramid_size, (3, 3), padding="SAME", name="fpn_p5")(p5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6 = kl.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(p5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        # RPN Model
        rpn = common.build_rpn_model(self.rpn_anchor_stride, len(self.rpn_anchor_ratios), self.top_down_pyramid_size)

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
            pass

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [kl.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = cfg.TRAIN.POST_NMS_ROIS if self.train_flag else cfg.TEST.POST_NMS_ROIS

        rpn_rois = common.ProposalLayer(proposal_count=proposal_count,
                                        nms_threshold=self.rpn_nms_threshold,
                                        batch_size=self.batch_size,
                                        name="ROI")([rpn_class, rpn_bbox, anchors])

        fc_layer_size = cfg.COMMON.FPN_CLASS_FC_LAYERS_SIZE
        pool_size = cfg.COMMON.POOL_SIZE
        mask_pool_size = cfg.COMMON.MASK_POOL_SIZE
        train_or_freeze = cfg.COMMON.TRAIN_FLAG

        if self.train_flag:

            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = kl.Lambda(lambda x: self.image_utils.parse_image_meta_graph(x)["active_class_ids"])(
                input_image_meta)

            if not cfg.TRAIN.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = kl.Input(shape=[proposal_count, 4], name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = kl.Lambda(lambda x: self.bbox_util.norm_boxes_graph(x, k.shape(input_image)[1:3]))(
                    input_rois)
            else:
                target_rois = rpn_rois
                input_rois = None

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask = \
                common.DetectionTargetLayer(self.batch_size, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = common.fpn_classifier_graph(rois,
                                                                                      mrcnn_feature_maps,
                                                                                      input_image_meta,
                                                                                      pool_size,
                                                                                      self.class_num,
                                                                                      train_flag=train_or_freeze,
                                                                                      fc_layers_size=fc_layer_size)

            mrcnn_mask = common.build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                     input_image_meta,
                                                     mask_pool_size,
                                                     self.class_num,
                                                     train_flag=train_or_freeze)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = kl.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = kl.Lambda(lambda x: common.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = kl.Lambda(lambda x: common.rpn_bbox_loss_graph(self.batch_size, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = kl.Lambda(lambda x: common.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = kl.Lambda(lambda x: common.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = kl.Lambda(lambda x: common.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]

            if not cfg.TRAIN.USE_RPN_ROIS:
                inputs.append(input_rois)

            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = km.Model(inputs, outputs, name='mask_rcnn')
            pass
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = common.fpn_classifier_graph(rpn_rois,
                                                                                      mrcnn_feature_maps,
                                                                                      input_image_meta,
                                                                                      pool_size,
                                                                                      self.class_num,
                                                                                      train_flag=train_or_freeze,
                                                                                      fc_layers_size=fc_layer_size)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = common.DetectionLayer(self.batch_size, name="mrcnn_detection")([rpn_rois,
                                                                                         mrcnn_class,
                                                                                         mrcnn_bbox,
                                                                                         input_image_meta])

            # Create masks for detections
            detection_boxes = kl.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = common.build_fpn_mask_graph(detection_boxes,
                                                     mrcnn_feature_maps,
                                                     input_image_meta,
                                                     mask_pool_size,
                                                     self.class_num,
                                                     train_flag=train_or_freeze)

            model = km.Model([input_image, input_image_meta, anchors],
                             [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')
            pass

        # Add multi-GPU support. 多 GPU 操作
        gpu_count = cfg.COMMON.GPU_COUNT
        if gpu_count > 1:
            from m_rcnn.parallel_model import ParallelModel
            model = ParallelModel(model, gpu_count)

        return model
        pass

    def load_weights(self, model_path, by_name=False, exclude=None):
        """
            Modified version of the corresponding Keras function
            with the addition of multi-GPU support and the ability
            to exclude some layers from loading.
        :param model_path:
        :param by_name:
        :param exclude: list of layer names to exclude
        :return:
        """

        if exclude:
            by_name = True
            pass

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
            pass

        model_file = h5py.File(model_path, mode='r')

        if 'layer_names' not in model_file.attrs and 'model_weights' in model_file:
            model_file = model_file['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model

        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers
        print("layers: {}".format(layers))

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(model_file, layers)
        else:
            saving.load_weights_from_hdf5_group(model_file, layers)
        if hasattr(model_file, 'close'):
            model_file.close()
        pass

    def generate_random_rois(self, image_shape, count, gt_boxes):
        """
            Generates ROI proposals similar to what a region proposal network
            would generate.
        :param image_shape: [Height, Width, Depth]
        :param count: Number of ROIs to generate
        :param gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.
        :return:
        """
        # placeholder
        rois = np.zeros((count, 4), dtype=np.int32)

        # Generate random ROIs around GT boxes (90% of count)
        rois_per_box = int(0.9 * count / gt_boxes.shape[0])
        for i in range(gt_boxes.shape[0]):
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
            h = gt_y2 - gt_y1
            w = gt_x2 - gt_x1
            # random boundaries
            r_y1 = max(gt_y1 - h, 0)
            r_y2 = min(gt_y2 + h, image_shape[0])
            r_x1 = max(gt_x1 - w, 0)
            r_x2 = min(gt_x2 + w, image_shape[1])

            # To avoid generating boxes with zero area, we generate double what
            # we need and filter out the extra. If we get fewer valid boxes
            # than we need, we loop and try again.
            while True:
                y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
                x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
                # Filter out zero area boxes
                threshold = 1
                y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:rois_per_box]
                x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:rois_per_box]
                if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                    break

            # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
            # into x1, y1, x2, y2 order
            x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
            y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
            box_rois = np.hstack([y1, x1, y2, x2])
            rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

        # Generate random ROIs anywhere in the image (10% of count)
        remaining_count = count - (rois_per_box * gt_boxes.shape[0])
        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
            x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >= threshold][:remaining_count]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >= threshold][:remaining_count]
            if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        global_rois = np.hstack([y1, x1, y2, x2])
        rois[-remaining_count:] = global_rois

        return rois
        pass

    def build_detection_targets(self, rpn_rois, gt_class_ids, gt_boxes, gt_masks):
        """
            Generate targets for training Stage 2 classifier and mask heads.
            This is not used in normal training. It's useful for debugging or to train
            the Mask RCNN heads without using the RPN head.
        :param rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
        :param gt_class_ids: [instance count] Integer class IDs
        :param gt_boxes: [instance count, (y1, x1, y2, x2)]
        :param gt_masks: [height, width, instance count] Ground truth masks. Can be full
                        size or mini-masks.
        :return:
            rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
            class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
            bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
                    bbox refinements.
            masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
                   to bbox boundaries and resized to neural network output size.
        """
        assert rpn_rois.shape[0] > 0
        assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
            gt_class_ids.dtype)
        assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
            gt_boxes.dtype)
        assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
            gt_masks.dtype)

        # It's common to add GT Boxes to ROIs but we don't do that here because
        # according to XinLei Chen's paper, it doesn't help.

        # Trim empty padding in gt_boxes and gt_masks parts
        instance_ids = np.where(gt_class_ids > 0)[0]
        assert instance_ids.shape[0] > 0, "Image must contain instances."
        gt_class_ids = gt_class_ids[instance_ids]
        gt_boxes = gt_boxes[instance_ids]
        gt_masks = gt_masks[:, :, instance_ids]

        # Compute areas of ROIs and ground truth boxes.
        # rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * (rpn_rois[:, 3] - rpn_rois[:, 1])
        # gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

        # Compute overlaps [rpn_rois, gt_boxes]
        overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
        for i in range(overlaps.shape[1]):
            gt = gt_boxes[i]
            overlaps[:, i] = self.bbox_util.compute_iou(gt, rpn_rois)
            pass

        # Assign ROIs to GT boxes
        rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
        rpn_roi_iou_max = overlaps[np.arange(overlaps.shape[0]), rpn_roi_iou_argmax]

        # GT box assigned to each ROI
        rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
        rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

        # Positive ROIs are those with >= 0.5 IoU with a GT box.
        fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

        # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
        # TODO: To hard example mine or not to hard example mine, that's the question
        # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
        bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

        # Subsample ROIs. Aim for 33% foreground.
        # FG
        fg_roi_count = int(self.rois_per_image * self.roi_positive_ratio)
        if fg_ids.shape[0] > fg_roi_count:
            keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
        else:
            keep_fg_ids = fg_ids
        # BG
        remaining = self.rois_per_image - keep_fg_ids.shape[0]
        if bg_ids.shape[0] > remaining:
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
        else:
            keep_bg_ids = bg_ids
        # Combine indices of ROIs to keep
        keep = np.concatenate([keep_fg_ids, keep_bg_ids])
        # Need more?
        remaining = self.rois_per_image - keep.shape[0]

        if remaining > 0:
            # Looks like we don't have enough samples to maintain the desired
            # balance. Reduce requirements and fill in the rest. This is
            # likely different from the Mask RCNN paper.

            # There is a small chance we have neither fg nor bg samples.
            if keep.shape[0] == 0:
                # Pick bg regions with easier IoU threshold
                bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
                assert bg_ids.shape[0] >= remaining
                keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
                assert keep_bg_ids.shape[0] == remaining
                keep = np.concatenate([keep, keep_bg_ids])
            else:
                # Fill the rest with repeated bg rois.
                keep_extra_ids = np.random.choice(
                    keep_bg_ids, remaining, replace=True)
                keep = np.concatenate([keep, keep_extra_ids])
        assert keep.shape[0] == self.rois_per_image, \
            "keep doesn't match ROI batch size {}, {}".format(keep.shape[0], self.rois_per_image)

        # Reset the gt boxes assigned to BG ROIs.
        rpn_roi_gt_boxes[keep_bg_ids, :] = 0
        rpn_roi_gt_class_ids[keep_bg_ids] = 0

        # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
        rois = rpn_rois[keep]
        roi_gt_boxes = rpn_roi_gt_boxes[keep]
        roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
        roi_gt_assignment = rpn_roi_iou_argmax[keep]

        # Class-aware bbox deltas. [y, x, log(h), log(w)]
        bboxes = np.zeros((self.rois_per_image, self.class_num, 4), dtype=np.float32)
        pos_ids = np.where(roi_gt_class_ids > 0)[0]
        bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = self.bbox_util.box_refinement(rois[pos_ids],
                                                                                   roi_gt_boxes[pos_ids, :4])
        # Normalize bbox refinements
        bbox_std_dev = np.array(cfg.COMMON.BBOX_STD_DEV)
        bboxes /= bbox_std_dev

        # Generate class-specific target masks
        masks = np.zeros((self.rois_per_image, self.image_shape[0], self.image_shape[1], self.class_num),
                         dtype=np.float32)

        for i in pos_ids:
            class_id = roi_gt_class_ids[i]
            assert class_id > 0, "class id must be greater than 0"
            gt_id = roi_gt_assignment[i]
            class_mask = gt_masks[:, :, gt_id]

            if cfg.TRAIN.USE_MINI_MASK:
                # Create a mask placeholder, the size of the image
                placeholder = np.zeros(self.image_shape[:2], dtype=bool)
                # GT box
                gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
                gt_w = gt_x2 - gt_x1
                gt_h = gt_y2 - gt_y1
                # Resize mini mask to size of GT box
                placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                    np.round(self.image_utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
                # Place the mini batch in the placeholder
                class_mask = placeholder

            # Pick part of the mask and resize it
            y1, x1, y2, x2 = rois[i].astype(np.int32)
            m = class_mask[y1:y2, x1:x2]
            mask = self.image_utils.resize(m, self.image_shape)
            masks[i, :, :, class_id] = mask

        return rois, roi_gt_class_ids, bboxes, masks
        pass

    # #############################################################################################
    # test
    # #############################################################################################

    def detect(self, images_info_list, verbose=0):
        """
            Runs the detection pipeline.
        :param images_info_list: List of images, potentially of different sizes.
        :param verbose:
        :return: a list of dicts, one dict per image. The dict contains:
            rois: [N, (y1, x1, y2, x2)] detection bounding boxes
            class_ids: [N] int class IDs
            scores: [N] float probability scores for the class IDs
            masks: [H, W, N] instance binary masks
        """
        if verbose:
            print("processing {} image_info.".format(len(images_info_list)))
            for image_info in images_info_list:
                print("image_info: {}".format(image_info))
                pass
            pass

        # Mold inputs to format expected by the neural network
        molded_images_list, image_metas_list, windows_list = self.image_utils.mode_input(images_info_list)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images_list[0].shape
        for g in molded_images_list[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
            pass

        # Anchors
        anchors = self.anchor_utils.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (cfg.TEST.BATCH_SIZE,) + anchors.shape)

        if verbose:
            print("molded_images_list: ", molded_images_list)
            print("image_metas_list: ", image_metas_list)
            print("anchors: ", anchors)
            pass

        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images_list, image_metas_list, anchors], verbose=0)
        # Process detections
        results_list = []
        for i, image_info in enumerate(images_info_list):
            molded_image_shape = molded_images_list[i].shape
            final_rois, final_class_ids, final_scores, final_masks = self.un_mold_detections(detections[i],
                                                                                             mrcnn_mask[i],
                                                                                             image_info.shape,
                                                                                             molded_image_shape,
                                                                                             windows_list[i])
            results_list.append({"rois": final_rois,
                                 "class_ids": final_class_ids,
                                 "scores": final_scores,
                                 "masks": final_masks,
                                 })
        return results_list
        pass

    def un_mold_detections(self, detections, mrcnn_mask, original_image_shape,
                           image_shape, window):
        """
            Reformats the detections of one image from the format of the neural
            network output to a format suitable for use in the rest of the
            application.
        :param detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        :param mrcnn_mask: [N, height, width, num_classes]
        :param original_image_shape: [H, W, C] Original image shape before resizing
        :param image_shape: [H, W, C] Shape of the image after resizing and padding
        :param window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                        image is excluding the padding.
        :return:
            boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            class_ids: [N] Integer class IDs for each bounding box
            scores: [N] Float probability scores of the class_id
            masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        n = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[: n, :4]
        class_ids = detections[: n, 4].astype(np.int32)
        scores = detections[: n, 5]
        masks = mrcnn_mask[np.arange(n), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = self.bbox_util.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = self.bbox_util.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            n = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(n):
            # Convert neural network mask to full size mask
            full_mask = self.mask_util.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
            pass
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks
        pass
