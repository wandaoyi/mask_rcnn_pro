#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/11 20:49
# @Author   : WanDaoYi
# @FileName : coco_dataset.py
# ============================================

from datetime import datetime
import os
import json
import itertools
import numpy as np
from collections import defaultdict
from config import cfg


def is_array_like(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class CocoDataset(object):

    def __init__(self, annotation_path, image_file_path):
        """
        :param annotation_path: annotation json path, as ./instances_train2014.json
        :param image_file_path: image file path, as ./train2014
        """

        # file path
        self.annotation_path = annotation_path
        self.image_file_path = image_file_path

        # dataset info
        self.dataset = self.read_coco_json_data()

        # class info
        self.categories_dict = self.categories_info()
        # image info
        self.image_dict = self.images_info()
        # annotations info, image to annotations info, class to image info
        self.annotations_dict, self.image_2_annotations, self.categories_2_image = self.annotations_info()

        self.image_info_list = []
        # Background is always the first class
        self.class_info_list = cfg.COMMON.DEFAULT_CLASS_INFO

        # 数据处理
        self.deal_data()

        # 类别数量
        self.class_num = len(self.class_info_list)
        # 类别 id list
        self.class_ids_list = np.arange(self.class_num)
        # 类别名字
        self.class_names_list = [self.clean_name(c["name"]) for c in self.class_info_list]
        # 图像数量
        self.images_num = len(self.image_info_list)
        # 图像 id list
        self._image_ids_list = np.arange(self.images_num)

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info_list]))
        self.source_class_ids = self.get_source_class_ids()

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): class_id
                                      for info, class_id in zip(self.class_info_list, self.class_ids_list)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): image_id
                                      for info, image_id in zip(self.image_info_list, self.image_ids_list)}

        pass

    @property
    def image_ids_list(self):
        return self._image_ids_list

    def get_source_class_ids(self):
        source_class_ids = {}
        # Loop over dataset
        for source in self.sources:
            source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info_list):
                # Include BG class in all dataset
                if i == 0 or source == info['source']:
                    source_class_ids[source].append(i)
                    pass
                pass
            pass

        return source_class_ids
        pass

    # load coco data
    def read_coco_json_data(self):

        # str to json
        print("json load.....")
        data_json = json.load(open(self.annotation_path, encoding="utf-8"))
        assert type(data_json) == dict, 'annotation file format {} not supported'.format(type(data_json))

        # json_key_list = [key_name for key_name in data_json]
        # print(json_key_list)

        return data_json

    # deal class info
    def categories_info(self):

        categories_dict = dict()
        if "categories" in self.dataset:
            print("categories info...")
            categories_info = self.dataset["categories"]
            for categories in categories_info:
                categories_dict[categories["id"]] = categories
                pass
            # categories_ids = [categories['id'] for categories in categories_info]
            # print(categories_ids)
            pass

        return categories_dict

    # deal image info
    def images_info(self):

        image_dict = dict()

        if "images" in self.dataset:
            print("images info...")
            image_info_list = self.dataset["images"]

            for image_info in image_info_list:
                image_dict[image_info["id"]] = image_info
                pass
            pass

        return image_dict

    # deal annotation info and image to annotation, class to image
    def annotations_info(self):

        annotations_dict = dict()
        image_2_annotations = defaultdict(list)
        categories_2_image = defaultdict(list)

        if "annotations" in self.dataset:
            print("annotations info...")
            annotations_list = self.dataset["annotations"]
            for annotations in annotations_list:
                annotations_dict[annotations["id"]] = annotations
                image_2_annotations[annotations["image_id"]].append(annotations)

                if "categories" in self.dataset:
                    categories_2_image[annotations["category_id"]].append(annotations["image_id"])
                    pass
                pass
            pass

        return annotations_dict, image_2_annotations, categories_2_image

    # image ids list
    def get_image_ids(self, image_ids=[], class_ids=[]):

        if len(image_ids) == 0 and len(class_ids) == 0:
            ids = self.image_dict.keys()
        else:
            ids = set(image_ids)
            for i, class_id in enumerate(class_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.categories_2_image[class_id])
                else:
                    ids &= set(self.categories_2_image[class_id])
        return list(ids)
        pass

    # class ids
    def get_class_ids(self):
        # class ids
        categories_ids = sorted([categories['id'] for categories in self.dataset['categories']])
        return categories_ids
        pass

    def get_annotation_ids(self, image_ids=[], class_ids=[], area_rang=[], is_crowd=False):
        """
        :param image_ids : (int array)  get annotation for given images
        :param class_ids: (int array)   get annotation for given classes
        :param area_rang: (float array) get annotation for given area range (e.g. [0 inf])
        :param is_crowd: (boolean)  get annotation for given crowd label (False or True)
        :return: annotation_ids: (int array)    integer array of ann ids
        """
        if len(image_ids) == len(class_ids) == len(area_rang) == 0:
            annotations = self.dataset['annotations']
            pass
        else:
            if len(image_ids) != 0:
                lists = [self.image_2_annotations[image_id] for image_id in image_ids if
                         image_id in self.image_2_annotations]
                annotations = list(itertools.chain.from_iterable(lists))
                pass
            else:
                annotations = self.dataset['annotations']
                pass
            annotations = annotations if len(class_ids) == 0 else [ann for ann in annotations if ann['category_id'] in class_ids]
            annotations = annotations if len(area_rang) == 0 else [ann for ann in annotations if ann['area'] > area_rang[0] and ann['area'] < area_rang[1]]

            pass
        if is_crowd:
            annotation_ids = [annotation['id'] for annotation in annotations if annotation['iscrowd'] == is_crowd]
            pass
        else:
            annotation_ids = [annotation['id'] for annotation in annotations]
            pass

        return annotation_ids
        pass

    def load_class_info(self, class_ids=[]):
        return [self.categories_dict[class_ids]]
        pass

    def load_annotation_info(self, annotation_ids=[]):

        if is_array_like(annotation_ids):
            return [self.annotations_dict[annotation_id] for annotation_id in annotation_ids]
        elif type(annotation_ids) == int:
            return [self.annotations_dict[annotation_ids]]

        pass

    # 增加类别信息
    def add_class(self, source, class_id, class_name):
        """
        :param source: 来源
        :param class_id: 类别的 id 号
        :param class_name: 类别名称
        :return:
        """

        assert "." not in source, "Source name cannot contain a dot"

        # 判断类别是否已存在
        for info_map in self.class_info_list:
            class_info_flag = info_map["source"] == source and info_map["id"] == class_id
            if class_info_flag:
                # source.class_id combination already available, skip
                return
                pass

        # 添加新的类别信息
        info_map = {"source": source,
                    "id": class_id,
                    "name": class_name
                    }
        self.class_info_list.append(info_map)
        pass

    # 添加图像信息
    def add_image(self, source, image_id, path, **kwargs):
        """
        :param source: 来源
        :param image_id: 图像 id
        :param path: 路径
        :param kwargs: 一个 map 超参
        :return:
        """
        image_info_map = {"id": image_id, "source": source, "path": path}
        image_info_map.update(kwargs)
        self.image_info_list.append(image_info_map)
        pass

    # 对数据处理
    def deal_data(self):

        image_ids = []
        class_ids = self.get_class_ids()
        for class_id in class_ids:
            image_ids.extend(list(self.get_image_ids(class_ids=[class_id])))
            pass
        # Remove duplicates
        image_ids = list(set(image_ids))

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, self.load_class_info(i)[0]["name"])
            pass

        # Add images
        for i in image_ids:
            self.add_image(source="coco", image_id=i,
                           path=os.path.join(self.image_file_path, self.image_dict[i]['file_name']),
                           width=self.image_dict[i]["width"],
                           height=self.image_dict[i]["height"],
                           annotations=self.load_annotation_info(self.get_annotation_ids(
                               image_ids=[i], class_ids=class_ids, is_crowd=False)))

        pass

    # class name value clean
    def clean_name(self, name):
        """
        :param name: name value
        :return:
        """
        return ",".join(name.split(",")[: 1])
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    anno_file_path = "G:/deep_learning_demo/data/instance_segmentation/annotations"
    train_anno_json_name = "instances_train2014.json"
    val_anno_json_name = "instances_minival2014.json"
    train_anno_json_data_path = os.path.join(anno_file_path, train_anno_json_name)
    val_anno_json_data_path = os.path.join(anno_file_path, val_anno_json_name)

    train_image_path = "G:/deep_learning_demo/data/instance_segmentation/train2014"
    val_image_path = "G:/deep_learning_demo/data/instance_segmentation/val2014"

    train_data = CocoDataset(train_anno_json_data_path, train_image_path)

    dataset = train_data.dataset
    dataset_key = [key for key in dataset]
    print("dataset_key: {}".format(dataset_key))
    print("dataset_type: {}".format(type(dataset)))
    print("info_type: {}".format(type(dataset["info"])))
    print("images_type: {}".format(type(dataset["images"])))
    print("licenses_type: {}".format(type(dataset["licenses"])))
    print("annotations_type: {}".format(type(dataset["annotations"])))
    print("categories_type: {}".format(type(dataset["categories"])))

    info_key = [key for key in dataset["info"]]
    print("info_key: {}".format(info_key))
    print("info: {}".format(dataset["info"]))
    print("licenses: {}".format(dataset["licenses"]))
    print("categories: {}".format(dataset["categories"]))
    print("images_0-1: {}".format(dataset["images"][: 2]))
    print("annotations_0-1: {}".format(dataset["annotations"][: 2]))

    print("It's over!")
    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
