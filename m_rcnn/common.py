#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/12 15:40
# @Author   : WanDaoYi
# @FileName : common.py
# ============================================

import numpy as np
import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM
import keras.engine as KE
from utils.misc_utils import MiscUtils
from utils.bbox_utils import BboxUtil
from utils.image_utils import ImageUtils
from config import cfg


def log2_graph(x):
    """
        Implementation of Log2. TF doesn't have a native implementation.
    """
    return tf.log(x) / tf.log(2.0)


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_flag=True):
    """
        conv_block is the block that has a conv layer at shortcut
    :param input_tensor: input tensor
    :param kernel_size: default 3, the kernel size of middle conv layer at main path
    :param filters: list of integers, the nb_filters of 3 conv layer at main path
    :param stage: integer, current stage label, used for generating layer names
    :param block: 'a','b'..., current block label, used for generating layer names
    :param strides:
    :param use_bias: Boolean. To use or not use a bias in conv layers.
    :param train_flag: Boolean. Train or freeze Batch Norm layers
    :return:
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_flag)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_flag)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_flag=True):
    """
        The identity_block is the block that has no conv layer at shortcut
    :param input_tensor: input tensor
    :param kernel_size: default 3, the kernel size of middle conv layer at main path
    :param filters: list of integers, the nb_filters of 3 conv layer at main path
    :param stage: nteger, current stage label, used for generating layer names
    :param block: 'a','b'..., current block label, used for generating layer names
    :param use_bias: Boolean. To use or not use a bias in conv layers.
    :param train_flag: Boolean. Train or freeze Batch Norm layers
    :return:
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_flag)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, class_num, train_flag=True):
    """
        Builds the computation graph of the mask head of Feature Pyramid Network.
    :param rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized coordinates.
    :param feature_maps: List of feature maps from different layers of the pyramid,
                        [P2, P3, P4, P5]. Each has a different resolution.
    :param image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    :param pool_size: The width of the square feature map generated from ROI Pooling.
    :param class_num: number of classes, which determines the depth of the results
    :param train_flag: Boolean. Train or freeze Batch Norm layers
    :return: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn1')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn2')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn3')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_mask_bn4')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(class_num, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)
    return x


def rpn_graph(feature_map, anchors_per_location, rpn_anchor_stride):
    """
        Builds the computation graph of Region Proposal Network.
    :param feature_map: backbone features [batch, height, width, depth]
    :param anchors_per_location: number of anchors per pixel in the feature map
    :param rpn_anchor_stride: Controls the density of anchors.
                            Typically 1 (anchors for every pixel in the feature map),
                            or 2 (every other pixel).
    :return:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=rpn_anchor_stride, name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]
    pass


def build_rpn_model(rpn_anchor_stride, anchors_per_location, depth):
    """
        Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared weights.
    :param rpn_anchor_stride: Controls the density of anchors.
                            Typically 1 (anchors for every pixel in the feature map),
                            or 2 (every other pixel).
    :param anchors_per_location: number of anchors per pixel in the feature map
    :param depth: Depth of the backbone feature map.
    :return: Returns a Keras Model object. The model outputs, when called, are:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh),
                  log(dw))] Deltas to be applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, rpn_anchor_stride)

    return KM.Model([input_feature_map], outputs, name="rpn_model")
    pass


def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size,
                         class_num, train_flag=True, fc_layers_size=1024):
    """
        Builds the computation graph of the feature pyramid network classifier
        and regressor heads.
    :param rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized coordinates.
    :param feature_maps: List of feature maps from different layers of the pyramid,
                        [P2, P3, P4, P5]. Each has a different resolution.
    :param image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    :param class_num: number of classes, which determines the depth of the results
    :param pool_size: The width of the square feature map generated from ROI Pooling.
    :param train_flag: Boolean. Train or freeze Batch Norm layers
    :param fc_layers_size: Size of the 2 FC layers
    :return:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    """

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)

    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_flag)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_flag)
    x = KL.Activation('relu')(x)

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(class_num),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(class_num * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], class_num, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox
    pass


def build_rpn_targets(anchors, gt_class_ids, gt_boxes):
    """
        Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.
    :param anchors: [num_anchors, (y1, x1, y2, x2)]
    :param gt_class_ids: [num_gt_boxes] Integer class IDs.
    :param gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
    :return:
        rpn_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_per_image = cfg.TRAIN.ANCHORS_PER_IMAGE
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((anchor_per_image, 4))

    bbox_util = BboxUtil()
    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = bbox_util.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
        pass
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)
        pass

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = bbox_util.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (anchor_per_image // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (anchor_per_image - np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
        pass

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox

    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox_std_dev = np.array(cfg.COMMON.RPN_BBOX_STD_DEV)
        rpn_bbox[ix] /= rpn_bbox_std_dev
        ix += 1

    return rpn_match, rpn_bbox
    pass


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """
        RPN anchor classifier loss.
    :param rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                      -1=negative, 0=neutral anchor.
    :param rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    :return:
    """

    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def batch_pack_graph(x, counts, num_rows):
    """
        Picks different number of values from each row in x depending on the values in counts.
    :param x:
    :param counts:
    :param num_rows:
    :return:
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def smooth_l1_loss(y_true, y_pred):
    """
        Implements Smooth-L1 loss. y_true and y_pred are typically: [N, 4], but could be any shape.
    :param y_true:
    :param y_pred:
    :return:
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_bbox_loss_graph(batch_size, target_bbox, rpn_match, rpn_bbox):
    """

    :param batch_size:
    :param target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
                        Uses 0 padding to fill in unsed bbox deltas.
    :param rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                      -1=negative, 0=neutral anchor.
    :param rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    :return: Return the RPN bounding box loss graph.
    """

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, batch_size)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids):
    """
        Loss for the classifier head of Mask RCNN.
    :param target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
                            padding to fill in the array.
    :param pred_class_logits: [batch, num_rois, num_classes]
    :param active_class_ids: [batch, num_classes]. Has a value of 1 for
                            classes that are in the dataset of the image, and 0
                            for classes that are not in the dataset.
    :return:
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """
        Loss for Mask R-CNN bounding box refinement.
    :param target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    :param target_class_ids: [batch, num_rois]. Integer class IDs.
    :param pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    :return:
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """
        Mask binary cross-entropy loss for the masks head.
    :param target_masks: [batch, num_rois, height, width].
                        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    :param target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    :param pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                        with values from 0 to 1.
    :return:
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def refine_detections_graph(rois, probs, deltas, window):
    """
        Refine classified proposals and filter overlaps and return final detections.
    :param rois: [N, (y1, x1, y2, x2)] in normalized coordinates
    :param probs: [N, num_classes]. Class probabilities.
    :param deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific bounding box deltas.
    :param window: (y1, x1, y2, x2) in normalized coordinates.
                   The part of the image that contains the image excluding the padding.
    :return: Returns detections shaped:
            [num_detections, (y1, x1, y2, x2, class_id, score)] where coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    bbox_utils = BboxUtil()
    refined_rois = bbox_utils.apply_box_deltas_graph(rois, deltas_specific * cfg.COMMON.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = bbox_utils.clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    defection_min_confidence = cfg.COMMON.DETECTION_MIN_CONFIDENCE
    if defection_min_confidence:
        conf_keep = tf.where(class_scores >= defection_min_confidence)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""

        defection_max_instances = cfg.TEST.DETECTION_MAX_INSTANCES
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(tf.gather(pre_nms_rois, ixs),
                                                  tf.gather(pre_nms_scores, ixs),
                                                  max_output_size=defection_max_instances,
                                                  iou_threshold=cfg.TEST.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = defection_max_instances - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([defection_max_instances])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = cfg.TEST.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([tf.gather(refined_rois, keep),
                            tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
                            tf.gather(class_scores, keep)[..., tf.newaxis]
                            ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = cfg.TEST.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


class ProposalLayer(KE.Layer):
    """
        Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinement deltas to anchors.

        Inputs:
            rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
            rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

        Returns:
            Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, batch_size, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)

        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size

        self.misc_utils = MiscUtils()
        self.bbox_utils = BboxUtil()

        pass

    def call(self, inputs):
        """
            这里的 call 方法，会被 __init__() 方法回调
        :param inputs:
        :return:
        """
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        rpn_bbox_std_dev = np.array(cfg.COMMON.RPN_BBOX_STD_DEV)
        deltas = deltas * np.reshape(rpn_bbox_std_dev, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(cfg.COMMON.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices

        scores = self.misc_utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                             self.batch_size)
        deltas = self.misc_utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                             self.batch_size)
        pre_nms_anchors = self.misc_utils.batch_slice([anchors, ix],
                                                      lambda a, x: tf.gather(a, x),
                                                      self.batch_size,
                                                      names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = self.misc_utils.batch_slice([pre_nms_anchors, deltas],
                                            lambda x, y: self.bbox_utils.apply_box_deltas_graph(x, y),
                                            self.batch_size,
                                            names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = self.misc_utils.batch_slice(boxes,
                                            lambda x: self.bbox_utils.clip_boxes_graph(x, window),
                                            self.batch_size,
                                            names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = self.misc_utils.batch_slice([boxes, scores], nms, self.batch_size)

        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


class DetectionTargetLayer(KE.Layer):
    """
        Subsamples proposals and generates target box refinement, class_ids, and masks for each.
        Inputs:
            proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
            gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
            gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                      coordinates.
            gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

        Returns: Target ROIs and corresponding class IDs, bounding box shifts, and masks.
            rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
                  coordinates
            target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
            target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
            target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                         Masks cropped to bbox boundaries and resized to neural
                         network output size.
        Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, batch_size, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.misc_utils = MiscUtils()

        self.rois_per_image = cfg.TRAIN.ROIS_PER_IMAGE
        self.mask_shape = cfg.TRAIN.MASK_SHAPE
        pass

    def call(self, inputs):
        """
            这里的 call 方法，会被 __init__() 方法回调
        :param inputs: 参数如下所示
        :return:
        """
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = self.misc_utils.batch_slice([proposals, gt_class_ids, gt_boxes, gt_masks],
                                              lambda w, x, y, z: self.misc_utils.detection_targets_graph(w, x, y, z),
                                              self.batch_size, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.rois_per_image, 4),  # rois
            (None, self.rois_per_image),  # class_ids
            (None, self.rois_per_image, 4),  # deltas
            (None, self.rois_per_image, self.mask_shape[0], self.mask_shape[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]


class PyramidROIAlign(KE.Layer):
    """
        Implements ROI Pooling on multiple levels of the feature pyramid.
        Inputs:
        - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                 coordinates. Possibly padded with zeros if not enough
                 boxes to fill the array.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - feature_maps: List of feature maps from different levels of the pyramid.
                        Each is [batch, height, width, channels]

        Output:
        Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

        self.image_utils = ImageUtils()

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = self.image_utils.parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)


class DetectionLayer(KE.Layer):
    """
        Takes classified proposal boxes and their bounding box deltas and
        returns the final detection boxes.
        Returns:
            [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
            coordinates are normalized.
    """

    def __init__(self, batch_size, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.detection_max_instances = cfg.TEST.DETECTION_MAX_INSTANCES
        self.image_utils = ImageUtils()
        self.bbox_utils = BboxUtil()
        self.misc_utils = MiscUtils()

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = self.image_utils.parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = self.bbox_utils.norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = self.misc_utils.batch_slice([rois, mrcnn_class, mrcnn_bbox, window],
                                                       lambda x, y, w, z: refine_detections_graph(x, y, w, z),
                                                       self.batch_size)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(detections_batch, [self.batch_size, self.detection_max_instances, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.detection_max_instances, 6)
