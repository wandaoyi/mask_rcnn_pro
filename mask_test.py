#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/18 14:42
# @Author   : WanDaoYi
# @FileName : mask_test.py
# ============================================

from datetime import datetime
import os
import colorsys
import skimage.io
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from m_rcnn.mask_rcnn import MaskRCNN
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from config import cfg


class MaskTest(object):

    def __init__(self):

        # 获取 类别 list
        self.class_names_path = cfg.COMMON.OUR_CLASS_NAMES_PATH
        self.class_names_list = self.read_class_name()

        # 测试图像的输入 和 输出 路径
        self.test_image_file_path = cfg.TEST.TEST_IMAGE_FILE_PATH
        self.output_image_path = cfg.TEST.OUTPUT_IMAGE_PATH

        # 加载网络模型
        self.mask_model = MaskRCNN(train_flag=False)
        # 加载权重模型
        self.mask_model.load_weights(cfg.TEST.COCO_MODEL_PATH, by_name=True)

        pass

    def read_class_name(self):
        with open(self.class_names_path, "r") as file:
            class_names_info = file.readlines()
            class_names_list = [class_names.strip() for class_names in class_names_info]

            return class_names_list
        pass

    def do_test(self, show_image_flag=False):
        """
            batch predict
        :param show_image_flag: show images or not
        :return:
        """
        test_image_name_list = os.listdir(self.test_image_file_path)

        for test_image_name in test_image_name_list:
            test_image_path = os.path.join(self.test_image_file_path, test_image_name)
            # 读取图像
            image_info = skimage.io.imread(test_image_path)

            # Run detection
            results_info_list = self.mask_model.detect([image_info])
            # print("results: {}".format(results_info_list))

            # Visualize results
            result_info = results_info_list[0]
            self.deal_instances(image_info, self.class_names_list, result_info)

            height, width = image_info.shape[:2]

            fig = plt.gcf()
            # 输出原始图像 width * height的像素
            fig.set_size_inches(width / 100.0, height / 100.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            # save images
            output_image_path = os.path.join(self.output_image_path, test_image_name)
            plt.savefig(output_image_path)

            if show_image_flag:
                plt.show()

            # clear a axis
            plt.cla()
            # will close all open figures
            plt.close("all")
            pass
        pass

    # 获取实例随机颜色
    def random_colors(self, n, bright=True):
        """
            Generate random colors. To get visually distinct colors, generate them in HSV space then
            convert to RGB.
        :param n: Number of instances
        :param bright: image bright
        :return:
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / n, 1, brightness) for i in range(n)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors
        pass

    # 给图像的实例添加 mask
    def apply_mask(self, image, mask, color, alpha=0.5):
        """
            Apply the given mask to the image.
        :param image:
        :param mask:
        :param color:
        :param alpha:
        :return:
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image
        pass

    def deal_instances(self, image_info, class_names_list, result_info,
                       fig_size=(7, 7), ax=None, show_mask=True, show_bbox=True,
                       colors=None, captions=None):
        """
            实例处理
        :param image_info: original image info
        :param class_names_list: list of class names of the dataset
        :param result_info:
                    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
                    masks: [height, width, num_instances]
                    class_ids: [num_instances]
                    scores: (optional) confidence scores for each box
        :param fig_size: (optional) the size of the image
        :param ax:
        :param show_mask: To show masks or not
        :param show_bbox: To show bounding boxes or not
        :param colors: (optional) An array or colors to use with each object
        :param captions: (optional) A list of strings to use as captions for each object
        :return:
        """
        # r = results[0]
        # visualize.display_instances(image_info, r['rois'], r['masks'], r['class_ids'],
        #                             self.class_names_list, r['scores'])
        boxes = result_info["rois"]
        masks = result_info["masks"]
        class_ids = result_info["class_ids"]
        scores = result_info["scores"]

        # Number of instances
        n = boxes.shape[0]
        if not n:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
            pass
        # Generate random colors
        colors = colors or self.random_colors(n)
        print("colors_len: {}".format(len(colors)))
        masked_image = image_info.astype(np.uint32).copy()

        if not ax:
            # fig_size 用来设置画布大小
            _, ax = plt.subplots(1, figsize=fig_size)
            pass

        # 不显示坐标
        ax.axis('off')

        for i in range(n):
            color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
                pass

            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                      alpha=0.7, linestyle="dashed",
                                      edgecolor=color, facecolor='none')
                ax.add_patch(p)
                pass

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names_list[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
                pass
            else:
                caption = captions[i]
                pass

            ax.text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="none")

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = self.apply_mask(masked_image, mask, color)
                pass

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)

            for flip in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                flip = np.fliplr(flip) - 1
                p = Polygon(flip, facecolor="none", edgecolor=color)
                ax.add_patch(p)
            pass

        masked_image_uint8 = masked_image.astype(np.uint8)

        # 将 masked_image_uint8 放入到 plt 中
        ax.imshow(masked_image_uint8)
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = MaskTest()
    # print(demo.class_names_list)
    demo.do_test()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))

