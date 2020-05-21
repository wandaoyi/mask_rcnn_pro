#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/15 17:31
# @Author   : WanDaoYi
# @FileName : mask_train.py
# ============================================

from datetime import datetime
import os
import re
import keras
import imgaug
import logging
import numpy as np
import multiprocessing
import tensorflow as tf
from m_rcnn import common
from utils.bbox_utils import BboxUtil
from m_rcnn.mask_rcnn import MaskRCNN
from utils.image_utils import ImageUtils
from utils.anchor_utils import AnchorUtils
from m_rcnn.coco_dataset import CocoDataset
from config import cfg


class MaskTrain(object):

    def __init__(self):

        self.anchor_utils = AnchorUtils()
        self.bbox_utils = BboxUtil()
        self.image_utils = ImageUtils()

        # 日志保存路径
        self.log_path = self.log_file_path(cfg.TRAIN.LOGS_PATH, cfg.TRAIN.DATA_SOURCE)
        # 模型保存路径
        self.model_save_path = cfg.TRAIN.SAVE_MODEL_PATH

        # 训练数据
        self.train_data = CocoDataset(cfg.TRAIN.COCO_TRAIN_ANN_PATH, cfg.TRAIN.COCO_TRAIN_IMAGE_PATH)
        # 验证数据
        self.val_data = CocoDataset(cfg.TRAIN.COCO_VAL_ANN_PATH, cfg.TRAIN.COCO_VAL_IMAGE_PATH)

        # 加载 mask 网络模型
        self.mask_model = MaskRCNN(train_flag=True)
        # 使用 原作者 1 + 80 类别的数据
        # self.mask_model.load_weights(cfg.TEST.COCO_MODEL_PATH, by_name=True)

        # 载入在MS COCO上的预训练模型, 跳过不一样的分类数目层
        self.mask_model.load_weights(cfg.TRAIN.MODEL_PATH, by_name=True,
                                     exclude=["mrcnn_class_logits",
                                              "mrcnn_bbox_fc",
                                              "mrcnn_bbox",
                                              "mrcnn_mask"])

        self.epoch = 0
        pass

    # 设置 日志文件夹
    def log_file_path(self, log_dir, data_source="coco"):
        log_start_time = datetime.now()
        log_file_name = "{}_{:%Y%m%dT%H%M}".format(data_source.lower(), log_start_time)
        log_path = os.path.join(log_dir, log_file_name)

        return log_path
        pass

    def do_mask_train(self):
        # image augmentation
        augmentation = imgaug.augmenters.Fliplr(0.5)

        print("training - stage 1")
        # training - stage 1
        self.train_details(self.train_data,
                           self.val_data,
                           learning_rate=cfg.TRAIN.ROUGH_LEARNING_RATE,
                           epochs=cfg.TRAIN.FIRST_STAGE_N_EPOCH,
                           layers=cfg.TRAIN.HEADS_LAYERS,
                           augmentation=augmentation)

        print("training - stage 2")
        # training - stage 2
        self.train_details(self.train_data, self.val_data,
                           learning_rate=cfg.TRAIN.ROUGH_LEARNING_RATE,
                           epochs=cfg.TRAIN.MIDDLE_STAGE_N_EPOCH,
                           layers=cfg.TRAIN.FOUR_MORE_LAYERS,
                           augmentation=augmentation)

        print("training - stage 3")
        # training - stage 3
        self.train_details(self.train_data, self.val_data,
                           learning_rate=cfg.TRAIN.FINE_LEARNING_RATE,
                           epochs=cfg.TRAIN.LAST_STAGE_N_EPOCH,
                           layers=cfg.TRAIN.ALL_LAYERS,
                           augmentation=augmentation)
        pass

    def train_details(self, train_data, val_data, learning_rate,
                      epochs, layers, augmentation=None,
                      custom_callbacks=None, no_augmentation_sources=None):
        """
            Train the model.
        :param train_data: Training data object
        :param val_data: val data object
        :param learning_rate: The learning rate to train with
        :param epochs: Number of training epochs. Note that previous training epochs
                        are considered to be done alreay, so this actually determines
                        the epochs to train in total rather than in this particaular
                        call.
        :param layers: Allows selecting wich layers to train. It can be:
                        - A regular expression to match layer names to train
                        - One of these predefined values:
                          heads: The RPN, classifier and mask heads of the network
                          all: All the layers
                          3+: Train Resnet stage 3 and up
                          4+: Train Resnet stage 4 and up
                          5+: Train Resnet stage 5 and up
        :param augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
                            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
                            flips images right/left 50% of the time. You can pass complex
                            augmentations as well. This augmentation applies 50% of the
                            time, and when it does it flips images right/left half the time
                            and adds a Gaussian blur with a random sigma in range 0 to 5.
        :param custom_callbacks: Optional. Add custom callbacks to be called
                                with the keras fit_generator method. Must be list of type keras.callbacks.
        :param no_augmentation_sources: Optional. List of sources to exclude for
                                        augmentation. A source is string that identifies a dataset and is
                                        defined in the Dataset class.
        :return:
        """

        # Pre-defined layer regular expressions
        layer_regex = cfg.TRAIN.LAYER_REGEX

        if layers in layer_regex:
            layers = layer_regex[layers]
            pass
        self.set_trainable(layers)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            pass

        # Callbacks
        callbacks = [keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=0,
                                                 write_graph=True, write_images=False),
                     keras.callbacks.ModelCheckpoint(self.model_save_path, verbose=0,
                                                     save_weights_only=True)]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks
            pass

        # Data generators
        train_generator = self.data_generator(train_data,
                                              augmentation=augmentation,
                                              batch_size=self.mask_model.batch_size,
                                              no_augmentation_sources=no_augmentation_sources)
        val_generator = self.data_generator(val_data, batch_size=self.mask_model.batch_size)

        self.compile(learning_rate, cfg.TRAIN.LEARNING_MOMENTUM)

        print("learning_rate: {}, checkpoint path: {}".format(learning_rate, self.model_save_path))

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
            pass
        else:
            workers = multiprocessing.cpu_count()
            pass

        self.mask_model.keras_model.fit_generator(generator=train_generator,
                                                  initial_epoch=self.epoch,
                                                  epochs=epochs,
                                                  steps_per_epoch=cfg.TRAIN.STEPS_PER_EPOCH,
                                                  callbacks=callbacks,
                                                  validation_data=val_generator,
                                                  validation_steps=cfg.TRAIN.VALIDATION_STEPS,
                                                  max_queue_size=100,
                                                  workers=workers,
                                                  use_multiprocessing=True,
                                                  )

        self.epoch = max(self.epoch, epochs)
        pass

    def data_generator(self, data, augmentation=None, batch_size=1, random_rois=0,
                       detection_targets=False, no_augmentation_sources=None):
        """
            A generator that returns images and corresponding target class ids,
            bounding box deltas, and masks.
        :param data: The Dataset object to pick data from
        :param augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
                            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
                            right/left 50% of the time.
        :param batch_size: How many images to return in each call
        :param random_rois: If > 0 then generate proposals to be used to train the
                             network classifier and mask heads. Useful if training
                             the Mask RCNN part without the RPN.
        :param detection_targets: If True, generate detection targets (class IDs, bbox
                                deltas, and masks). Typically for debugging or visualizations because
                                in trainig detection targets are generated by DetectionTargetLayer.
        :param no_augmentation_sources: Optional. List of sources to exclude for
                                        augmentation. A source is string that identifies a dataset and is
                                        defined in the Dataset class.
        :return: Returns a Python generator. Upon calling next() on it, the
                generator returns two lists, inputs and outputs. The contents
                of the lists differs depending on the received arguments:
            inputs list:
                - images: [batch, H, W, C]
                - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
                - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
                - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
                - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
                - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
                - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                            are those of the image unless use_mini_mask is True, in which
                            case they are defined in MINI_MASK_SHAPE.

            outputs list: Usually empty in regular training. But if detection_targets
                        is True then the outputs list contains target class_ids, bbox deltas,
                        and masks.
        """
        # batch item index
        batch_index = 0
        image_index = -1
        image_ids = np.copy(data.image_ids_list)
        error_count = 0
        no_augmentation_sources = no_augmentation_sources or []

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        # Generate Anchors
        anchors = self.anchor_utils.generate_pyramid_anchors(image_shape=cfg.COMMON.IMAGE_SHAPE)

        image_id = ""

        mini_mask = cfg.TRAIN.USE_MINI_MASK
        max_gt_instances = cfg.TRAIN.MAX_GT_INSTANCES
        mean_pixel = np.array(cfg.COMMON.MEAN_PIXEL)

        # Keras requires a generator to run indefinitely.
        while True:
            try:
                # Increment index to pick next image. Shuffle if at the start of an epoch.
                image_index = (image_index + 1) % len(image_ids)
                if image_index == 0:
                    np.random.shuffle(image_ids)

                # Get GT bounding boxes and masks for image.
                image_id = image_ids[image_index]

                # If the image source is not to be augmented pass None as augmentation
                if data.image_info_list[image_id]['source'] in no_augmentation_sources:

                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = self.bbox_utils.load_image_gt(data,
                                                                                                        image_id,
                                                                                                        None,
                                                                                                        mini_mask)
                else:
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = self.bbox_utils.load_image_gt(data,
                                                                                                        image_id,
                                                                                                        augmentation,
                                                                                                        mini_mask)

                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                if not np.any(gt_class_ids > 0):
                    continue
                    pass

                # RPN Targets
                rpn_match, rpn_bbox = common.build_rpn_targets(anchors, gt_class_ids, gt_boxes)

                # 在这里定义 变量，避免下面使用的时候出现未定义
                rpn_rois = None
                rois = None
                mrcnn_class_ids = None
                mrcnn_bbox = None
                mrcnn_mask = None

                # Mask R-CNN Targets
                if random_rois:
                    rpn_rois = self.mask_model.generate_random_rois(image.shape, random_rois, gt_boxes)

                    if detection_targets:
                        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                            self.mask_model.build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks)
                        pass
                    pass

                # Init batch arrays
                if batch_index == 0:
                    batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros([batch_size, cfg.TRAIN.ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                    batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros((batch_size, max_gt_instances), dtype=np.int32)
                    batch_gt_boxes = np.zeros((batch_size, max_gt_instances, 4), dtype=np.int32)
                    batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1],
                                               max_gt_instances), dtype=gt_masks.dtype)

                    if random_rois:
                        batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)

                        if detection_targets:
                            batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
                            batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape,
                                                             dtype=mrcnn_class_ids.dtype)
                            batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                            batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)
                            pass
                        pass
                    pass

                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > max_gt_instances:
                    ids = np.random.choice(
                        np.arange(gt_boxes.shape[0]), max_gt_instances, replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]

                # Add to batch
                batch_image_meta[batch_index] = image_meta
                batch_rpn_match[batch_index] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[batch_index] = rpn_bbox
                batch_images[batch_index] = self.image_utils.mold_image(image.astype(np.float32), mean_pixel)
                batch_gt_class_ids[batch_index, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[batch_index, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[batch_index, :, :, :gt_masks.shape[-1]] = gt_masks

                if random_rois:
                    batch_rpn_rois[batch_index] = rpn_rois
                    if detection_targets:
                        batch_rois[batch_index] = rois
                        batch_mrcnn_class_ids[batch_index] = mrcnn_class_ids
                        batch_mrcnn_bbox[batch_index] = mrcnn_bbox
                        batch_mrcnn_mask[batch_index] = mrcnn_mask
                        pass
                    pass
                batch_index += 1

                # Batch full?
                if batch_index >= batch_size:
                    inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                              batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                    outputs = []

                    if random_rois:
                        inputs.extend([batch_rpn_rois])
                        if detection_targets:
                            inputs.extend([batch_rois])
                            # Keras requires that output and targets have the same number of dimensions
                            batch_mrcnn_class_ids = np.expand_dims(
                                batch_mrcnn_class_ids, -1)
                            outputs.extend(
                                [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                    yield inputs, outputs

                    # start a new batch
                    batch_index = 0
                pass
            except (GeneratorExit, KeyboardInterrupt):
                raise
            except:
                # Log it and skip the image
                logging.exception("Error processing image {}".format(data.image_info_list[image_id]))
                error_count += 1
                if error_count > 5:
                    raise

            pass
        pass

    def set_trainable(self, layer_regex, mask_model=None, indent=0, verbose=1):
        """
            Sets model layers as trainable if their names match
            the given regular expression.
        :param layer_regex:
        :param mask_model:
        :param indent:
        :param verbose:
        :return:
        """

        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and mask_model is None:
            print("Selecting layers to train")
            pass

        mask_model = mask_model or self.mask_model.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = mask_model.inner_model.layers if hasattr(mask_model, "inner_model") else mask_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, mask_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))

            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable

            # Print trainable layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))
        pass

    def compile(self, learning_rate, momentum_param):
        """
            Gets the model ready for training. Adds losses, regularization, and
            metrics. Then calls the Keras compile() function.
        :param learning_rate:
        :param momentum_param:
        :return:
        """

        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum_param,
                                         clipnorm=cfg.TRAIN.GRADIENT_CLIP_NORM)

        self.mask_model.keras_model._losses = []
        self.mask_model.keras_model._per_input_losses = {}

        loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                      "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]

        for name in loss_names:
            layer = self.mask_model.keras_model.get_layer(name)
            if layer.output in self.mask_model.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output, keepdims=True) * cfg.COMMON.LOSS_WEIGHTS.get(name, 1.))
            self.mask_model.keras_model.add_loss(loss)
            pass

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(cfg.TRAIN.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.mask_model.keras_model.trainable_weights if
                      'gamma' not in w.name and 'beta' not in w.name]

        self.mask_model.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.mask_model.keras_model.compile(optimizer=optimizer, loss=[None] * len(self.mask_model.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.mask_model.keras_model.metrics_names:
                continue
                pass
            layer = self.mask_model.keras_model.get_layer(name)

            self.mask_model.keras_model.metrics_names.append(name)

            loss = (tf.reduce_mean(layer.output, keepdims=True) * cfg.COMMON.LOSS_WEIGHTS.get(name, 1.))

            self.mask_model.keras_model.metrics_tensors.append(loss)
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = MaskTrain()
    demo.do_mask_train()
    print("hello world! ")

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
