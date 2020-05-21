#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/13 22:57
# @Author   : WanDaoYi
# @FileName : image_utils.py
# ============================================

import numpy as np
import skimage.color
import skimage.io
import skimage.transform
from distutils.version import LooseVersion
from config import cfg


class ImageUtils(object):

    def __init__(self):
        self.mean_pixel = np.array(cfg.COMMON.MEAN_PIXEL)
        pass

    def parse_image_meta_graph(self, meta):
        """
            Parses a tensor that contains image attributes to its components.
            See compose_image_meta() for more details.
        :param meta: [batch, meta length] where meta length depends on NUM_CLASSES
        :return: Returns a dict of the parsed tensors.
        """

        image_id = meta[:, 0]
        original_image_shape = meta[:, 1:4]
        image_shape = meta[:, 4:7]
        window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
        scale = meta[:, 11]
        active_class_ids = meta[:, 12:]
        return {
            "image_id": image_id,
            "original_image_shape": original_image_shape,
            "image_shape": image_shape,
            "window": window,
            "scale": scale,
            "active_class_ids": active_class_ids,
        }
        pass

    def compose_image_meta(self, image_id, original_image_shape, image_shape,
                           window, scale, active_class_ids):
        """
            Takes attributes of an image and puts them in one 1D array.
        :param image_id: An int ID of the image. Useful for debugging.
        :param original_image_shape: [H, W, C] before resizing or padding.
        :param image_shape: [H, W, C] after resizing and padding
        :param window: (y1, x1, y2, x2) in pixels. The area of the image where the real
                        image is (excluding the padding)
        :param scale: The scaling factor applied to the original image (float32)
        :param active_class_ids: List of class_ids available in the dataset from which
                                the image came. Useful if training on images from multiple datasets
                                where not all classes are present in all datasets.
        :return:
        """

        meta = np.array([image_id] +  # size=1
                        list(original_image_shape) +  # size=3
                        list(image_shape) +  # size=3
                        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
                        [scale] +  # size=1
                        list(active_class_ids)  # size=class_num
                        )
        return meta
        pass

    def load_image(self, image_path):
        """
            Load the specified image and return a [H,W,3] Numpy array.
        :param image_path: image path
        :return:
        """
        # Load image
        image = skimage.io.imread(image_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
        pass

    def mold_image(self, images, mean_pixel):
        """
            Expects an RGB image (or array of images) and subtracts
            the mean pixel and converts it to float. Expects image
            colors in RGB order.
        :param images:
        :param mean_pixel:
        :return:
        """
        return images.astype(np.float32) - np.array(mean_pixel)
        pass

    def mode_input(self, images_info_list):
        """
            Takes a list of images and modifies them to the format expected
            as an input to the neural network.
        :param images_info_list: List of image matrices [height,width,depth]. Images can have
                                different sizes.
        :return: returns 3 Numpy matrices:
            molded_images_list: [N, h, w, 3]. Images resized and normalized.
            image_metas_list: [N, length of meta data]. Details about each image.
            windows_list: [N, (y1, x1, y2, x2)]. The portion of the image that has the
                        original image (padding excluded).
        """

        molded_images_list = []
        image_metas_list = []
        windows_list = []

        image_mi_dim = cfg.COMMON.IMAGE_MIN_DIM
        image_max_dim = cfg.COMMON.IMAGE_MAX_DIM
        image_min_scale = cfg.COMMON.IMAGE_MIN_SCALE
        image_resize_mode = cfg.COMMON.IMAGE_RESIZE_MODE

        for image_info in images_info_list:
            # resize image
            molded_image, window, scale, padding, crop = self.resize_image(image_info,
                                                                           min_dim=image_mi_dim,
                                                                           min_scale=image_min_scale,
                                                                           max_dim=image_max_dim,
                                                                           resize_mode=image_resize_mode)

            molded_image = self.mold_image(molded_image, self.mean_pixel)

            # Build image_meta
            image_meta = self.compose_image_meta(0, image_info.shape, molded_image.shape, window, scale,
                                                 np.zeros([cfg.COMMON.CLASS_NUM], dtype=np.int32))
            # Append
            molded_images_list.append(molded_image)
            image_metas_list.append(image_meta)
            windows_list.append(window)
            pass

        # Pack into arrays
        molded_images_list = np.stack(molded_images_list)
        image_metas_list = np.stack(image_metas_list)
        windows_list = np.stack(windows_list)
        return molded_images_list, image_metas_list, windows_list
        pass

    def resize(self, image, output_shape, order=1, resize_mode="constant", cval=0, clip=True,
               preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
        """
            A wrapper for Scikit-Image resize().
            Scikit-Image generates warnings on every call to resize() if it doesn't
            receive the right parameters. The right parameters depend on the version
            of skimage. This solves the problem by using different parameters per
            version. And it provides a central place to control resizing defaults.
        :param image:
        :param output_shape:
        :param order:
        :param resize_mode:
        :param cval:
        :param clip:
        :param preserve_range:
        :param anti_aliasing:
        :param anti_aliasing_sigma:
        :return:
        """
        if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
            # New in 0.14: anti_aliasing. Default it to False for backward
            # compatibility with skimage 0.13.
            return skimage.transform.resize(image, output_shape,
                                            order=order, mode=resize_mode, cval=cval, clip=clip,
                                            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
                                            anti_aliasing_sigma=anti_aliasing_sigma)
        else:
            return skimage.transform.resize(image, output_shape,
                                            order=order, mode=resize_mode, cval=cval, clip=clip,
                                            preserve_range=preserve_range)
        pass

    def resize_image(self, image, min_dim=None, max_dim=None, min_scale=None, resize_mode="square"):
        """
            resize an image keeping the aspect ratio unchanged.
        :param image:
        :param min_dim: if provided, resize the image such that it's smaller dimension == min_dim
        :param max_dim: if provided, ensures that the image longest side doesn't
                        exceed this value.
        :param min_scale: if provided, ensure that the image is scaled up by at least
                          this percent even if min_dim doesn't require it.
        :param resize_mode: resizing mode.
                none: No resizing. Return the image unchanged.
                square: Resize and pad with zeros to get a square image
                    of size [max_dim, max_dim].
                pad64: Pads width and height with zeros to make them multiples of 64.
                       If min_dim or min_scale are provided, it scales the image up
                       before padding. max_dim is ignored in this mode.
                       The multiple of 64 is needed to ensure smooth scaling of feature
                       maps up and down the 6 levels of the FPN pyramid (2**6=64).
                crop: Picks random crops from the image. First, scales the image based
                      on min_dim and min_scale, then picks a random crop of
                      size min_dim x min_dim. Can be used in training only.
                      max_dim is not used in this mode.
        :return:
            image: the resized image
            window: (y1, x1, y2, x2). If max_dim is provided, padding might
                    be inserted in the returned image. If so, this window is the
                    coordinates of the image part of the full image (excluding
                    the padding). The x2, y2 pixels are not included.
            scale: The scale factor used to resize the image
            padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
        """
        # Keep track of image dtype and return results in the same dtype
        image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]
        crop = None

        if resize_mode == "none":
            return image, window, scale, padding, crop
            pass

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
            pass
        if min_scale and scale < min_scale:
            scale = min_scale
            pass

        # Does it exceed max dim?
        if max_dim and resize_mode == "square":
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max
                pass
            pass

        # Resize image using bilinear interpolation
        if scale != 1:
            image = self.resize(image, (round(h * scale), round(w * scale)), preserve_range=True)
            pass

        # Need padding or cropping?
        if resize_mode == "square":
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
            pass

        elif resize_mode == "pad64":
            h, w = image.shape[:2]
            # Both sides must be divisible by 64
            assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            # Height
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            # Width
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
            pass

        elif resize_mode == "crop":
            # Pick a random crop
            h, w = image.shape[:2]
            y = np.random.randint(0, (h - min_dim))
            x = np.random.randint(0, (w - min_dim))
            crop = (y, x, min_dim, min_dim)
            image = image[y:y + min_dim, x:x + min_dim]
            window = (0, 0, min_dim, min_dim)
            pass

        else:
            raise Exception("Mode {} not supported".format(resize_mode))
            pass

        return image.astype(image_dtype), window, scale, padding, crop

        pass
