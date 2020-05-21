# Mask R-CNN for Object Detection and Segmentation
# [mask_rcnn_pro](https://github.com/wandaoyi/mask_rcnn_pro)

- [论文地址](https://arxiv.org/abs/1703.06870)
- [我的 CSDN 博客](https://blog.csdn.net/qq_38299170/article/details/105233638) 
本项目使用 python3, keras 和 tensorflow 相结合。本模型基于 FPN 网络 和 resNet101 背骨，对图像中的每个目标生成 bounding boxes 和 分割 masks。

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset

```bashrc
mask_rcnn_coco.h5:
- https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

train2014 data:
- http://images.cocodataset.org/zips/train2014.zip
- http://images.cocodataset.org/annotations/annotations_trainval2014.zip

val2014 data(valminusminival):
- http://images.cocodataset.org/zips/val2014.zip
- https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0

val2014 data(minival):
- http://images.cocodataset.org/zips/val2014.zip
- https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0
```


# Getting Started

* 参考 config.py 文件配置。
* 下面文件中 def __init__(self) 方法中的配置文件，基本都是来自于 config.py


* 测试看效果:
* mask_test.py 下载好 mask_rcnn_coco.h5 模型，随便找点数据，设置好配置文件，直接运行看结果吧。


* 数据处理:
* prepare.py 直接运行代码，将 labelme json 数据制作成 coco json 数据。
* 并将数据进行划分


* 数据训练:
* mask_train.py 直接运行代码，观察 loss 情况。
* mask_rcnn_coco.h5 作为预训练模型很强，训练模型会有一个很好的起点。


* 多 GPU 训练:
* parallel_model.py: 本人没有多 GPU，这一步没做到验证,里面的代码，是沿用作者的。


* 本项目，操作上，就三板斧搞定，不搞那么复杂，吓到人。



