#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/21 15:53
# @Author   : WanDaoYi
# @FileName : prepare.py
# ============================================

"""
从 coco 的 json 数据中，得到一张图片的标注信息如下，包含5大部分的字段信息:

    "info"的value是一个dict，存储数据集的一些基本信息，我们不需要关注；

    "licenses"的value是一个list，存储license信息，我们不需要关注；

    "categories"的value是一个list，存储数据集的类别信息，包括类别的超类、类别id、类别名称；

    “images”的value是一个list，存储这张图片的基本信息，包括图片名、长、宽、id等重要信息；

    "annotations"的value是一个list，存储这张图片的标注信息，非常重要，list中的每一个元素是一个dict，
                    也即一个标注对象（instance）的信息。包括的字段有"segmentation"：标注点的坐标，
                    从第一个的x,y坐标一直到最后一个点的x,y坐标；
                    "area"是标注的闭合多边形的面积；
                    "iscrowd"表示对象之间是否有重叠; 0 表示不重叠
                    "image_id"是图片的id；
                    "bbox"是instance的边界框的左上角的x,y，边界框的宽和高；
                    "category_id"是这个instance对应的类别id；
                    "id"表示此instance标注信息在所有instance标注信息中的id。
"""


from datetime import datetime
import os
import json
import random
import numpy as np
from config import cfg


class Prepare(object):

    def __init__(self):

        self.label_me_json_file_path = "./dataset/ann_json"
        self.ori_image_file_path = "./dataset/images"
        self.save_data_path = "./infos"

        self.cate_and_super = self.load_json_data("./infos/cate_and_super.json")
        # 默认 BG 为背景 class name
        self.class_name_list = self.load_txt_data("./infos/our_class_names.txt")

        # 数据的百分比
        self.test_percent = cfg.COMMON.TEST_PERCENT
        self.val_percent = cfg.COMMON.VAL_PERCENT

        # 各成分数据保存路径
        self.train_data_path = cfg.COMMON.TRAIN_DATA_PATH
        self.val_data_path = cfg.COMMON.VAL_DATA_PATH
        self.test_data_path = cfg.COMMON.TEST_DATA_PATH

        self.train_image_name_list = []
        self.val_image_name_list = []

        # info 和 licenses 基本是固定的，所以可以在这里写死。
        # 具体信息你想怎么写就怎么写，感觉，无关痛痒。如果是要做记录，则需要写好点而已。
        self.info = {"description": "our data", "url": "",
                     "version": "1.0", "year": 2020,
                     "contributor": "our leader",
                     "date_created": "2020/05/20"}
        self.licenses = [{'url': "", 'id': 1, 'name': 'our leader'}]
        self.categories = self.category_info()
        self.images = []
        self.annotations = []

        self.ann_id = 0

        pass

    def load_txt_data(self, file_path):
        with open(file_path, encoding="utf-8") as file:
            data_info = file.readlines()
            data_list = [data.strip() for data in data_info]
            return data_list
            pass
        pass

    def load_json_data(self, file_path):
        with open(file_path, encoding="utf-8") as file:
            return json.load(file)
            pass
        pass

    def divide_data(self):
        """
            train, val, test 数据划分
        :return:
        """
        # 原始图像名字的 list
        image_name_list = os.listdir(self.ori_image_file_path)
        # 统计有多少张图像
        image_number = len(image_name_list)

        # 根据百分比得到各成分 数据量
        n_test = int(image_number * self.test_percent)
        n_val = int(image_number * self.val_percent)
        n_train = image_number - n_test - n_val

        if os.path.exists(self.train_data_path):
            os.remove(self.train_data_path)
            pass

        if os.path.exists(self.val_data_path):
            os.remove(self.val_data_path)
            pass

        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
            pass

        # 随机划分数据
        n_train_val = n_train + n_val
        train_val_list = random.sample(image_name_list, n_train_val)
        train_list = random.sample(train_val_list, n_train)

        train_file = open(self.train_data_path, "w")
        val_file = open(self.val_data_path, "w")
        test_file = open(self.test_data_path, "w")

        for image_name in image_name_list:
            if image_name in train_val_list:
                if image_name in train_list:
                    # 将训练的数据名称放到 list 中，不用再次去读写。
                    self.train_image_name_list.append(image_name)
                    # 将训练数据保存下来，可以用来参考，后续代码中不用到这个文件
                    train_file.write(image_name + "\n")
                    pass
                else:
                    # 将验证的数据名称放到 list 中，不用再次去读写。
                    self.val_image_name_list.append(image_name)
                    # 将验证数据保存下来，可以用来参考，后续代码中不用到这个文件
                    val_file.write(image_name + "\n")
                    pass
                pass
            else:
                # 测试图像，这个可以在 mask_test.py 文件中用于 test
                test_file.write(image_name + "\n")
                pass
            pass

        train_file.close()
        val_file.close()
        test_file.close()
        pass

    def category_info(self):
        categories = []
        class_name_list_len = len(self.class_name_list)
        for i in range(1, class_name_list_len):
            category_info = {}
            class_name = self.class_name_list[i]
            super_cate = self.cate_and_super[class_name]
            category_info.update({"supercategory": super_cate})
            category_info.update({"id": i})
            category_info.update({"name": class_name})

            categories.append(category_info)
            pass
        return categories
        pass

    def json_dump(self, data_info, file_path):

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data_info, file, ensure_ascii=False, indent=2)
        pass

    def coco_data_info(self):
        data_info = {}
        data_info.update({"info": self.info})
        data_info.update({"license": self.licenses})
        data_info.update({"categories": self.categories})
        data_info.update({"images": self.images})
        data_info.update({"annotations": self.annotations})
        return data_info
        pass

    def do_data_2_coco(self):
        # 划分数据
        self.divide_data()

        # 将划分的训练数据做成 coco 数据
        for train_image_name in self.train_image_name_list:
            name_info = train_image_name.split(".")[0]

            ann_json_name = name_info + ".json"
            ann_json_path = os.path.join(self.label_me_json_file_path, ann_json_name)
            json_data = self.load_json_data(ann_json_path)

            self.image_info(json_data, name_info, train_image_name)
            self.annotation_info(json_data, name_info)

            pass
        train_data = self.coco_data_info()
        train_data_path = os.path.join(self.save_data_path, "train_data.json")
        self.json_dump(train_data, train_data_path)

        # 初始化，不受上面训练数据影响
        self.images = []
        self.annotations = []

        # 将划分的验证数据做成 coco 数据
        for val_image_name in self.val_image_name_list:
            name_info = val_image_name.split(".")[0]

            ann_json_name = name_info + ".json"
            ann_json_path = os.path.join(self.label_me_json_file_path, ann_json_name)
            json_data = self.load_json_data(ann_json_path)

            self.image_info(json_data, name_info, val_image_name)
            self.annotation_info(json_data, name_info)

            pass

        val_data = self.coco_data_info()
        val_data_path = os.path.join(self.save_data_path, "val_data.json")
        self.json_dump(val_data, val_data_path)

        pass

    def image_info(self, json_data, name_info, train_image_name):

        image_info = {}
        height = json_data["imageHeight"]
        width = json_data["imageWidth"]

        image_info.update({"height": height})
        image_info.update({"width": width})
        image_info.update({"id": int(name_info)})
        image_info.update({"file_name": train_image_name})
        self.images.append(image_info)
        pass

    def annotation_info(self, json_data, name_info):

        data_shape = json_data["shapes"]
        for shape_info in data_shape:
            annotation = {}

            label = shape_info["label"]
            points = shape_info["points"]
            category_id = self.class_name_list.index(label)

            annotation.update({"id": self.ann_id})
            annotation.update({"image_id": int(name_info)})
            annotation.update({"category_id": category_id})
            segmentation = [np.asarray(points).flatten().tolist()]
            annotation.update({"segmentation": segmentation})
            bbox = self.bounding_box_info(points)
            annotation.update({"bbox": bbox})
            annotation.update({"iscrowd": 0})
            area = annotation['bbox'][-1] * annotation['bbox'][-2]
            annotation.update({"area": area})

            self.annotations.append(annotation)
            self.ann_id += 1
        pass

    def bounding_box_info(self, points):
        """
            # COCO的格式： [x1, y1, w, h] 对应COCO的bbox格式
        :param points: "points": [[160.0, 58.985], [151.2, 60.1], ..., [166.1, 56.1]]
        :return:
        """
        # np.inf 为无穷大
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = Prepare()
    demo.do_data_2_coco()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass


