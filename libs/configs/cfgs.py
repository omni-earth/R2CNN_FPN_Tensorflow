# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os

"""
v4-dense_feature_pyramid, v5-feature_pyramid
Test(快速训练):
v4
*********horizontal eval*********
水平， 2000张  3000 nms
R: 0.803773584906
P: 0.963236661407
mAP: 0.776797872684
F: 0.876309784923

***********rotate eval***********
旋转， 2000张  3000 nms
R: 0.769496855346
P: 0.89145996997
mAP: 0.690753310464
F: 0.826000571602

水平标准：
*********horizontal eval*********
水平， 2000张  3000 nms iou 0.4
R: 0.806289308176
P: 0.967120965901
mAP: 0.781793358603
F: 0.879412176548
水平， 2000张  3000 nms iou 0.5
R: 0.803773584906
P: 0.963236661407
mAP: 0.776797872684
F: 0.876309784923
水平， 2000张  3000 nms iou 0.6
R: 0.792452830189
P: 0.951222839664
mAP: 0.756438964669
F: 0.864609450559
水平， 2000张  3000 nms iou 0.7
R: 0.751257861635
P: 0.893903438016
mAP: 0.679421925197
F: 0.816396526583
***********rotate eval***********
旋转， 2000张  3000 nms  iou 0.4
R: 0.817924528302
P: 0.942568041236
mAP: 0.775231905212
F: 0.87583388179
旋转， 2000张  3000 nms  iou 0.5
R: 0.808805031447
P: 0.934789235475
mAP: 0.760185750655
F: 0.867245610218
旋转， 2000张  3000 nms  iou 0.6
R: 0.790880503145
P: 0.915325169604
mAP: 0.728443572557
F: 0.848564557298
旋转， 2000张  3000 nms  iou 0.7
R: 0.730188679245
P: 0.844232716838
mAP: 0.62203781869
F: 0.783080278275

v5
*********horizontal eval*********
水平， 2000张  3000 nms
R: 0.81320754717
P: 0.962663777256
mAP: 0.785794751386
F: 0.881646590363
***********rotate eval***********
旋转， 2000张  3000 nms
R: 0.786163522013
P: 0.885101755072
mAP: 0.700995138344
F: 0.832704086715




水平标准：
*********horizontal eval*********
水平， 2000张  3000 nms  iou 0.4
R: 0.814150943396
P: 0.962856197701
mAP: 0.786882346129
F: 0.882281521085
水平， 2000张  3000 nms  iou 0.5
R: 0.81320754717
P: 0.962663777256
mAP: 0.785794751386
F: 0.881646590363
水平， 2000张  3000 nms  iou 0.6
R: 0.802201257862
P: 0.949192944629
mAP: 0.76520138242
F: 0.869528713812
水平， 2000张  3000 nms  iou 0.7
R: 0.757547169811
P: 0.900415522966
mAP: 0.687343366523
F: 0.822825789806
***********rotate eval***********
旋转， 2000张  3000 nms  iou 0.4
R: 0.83679245283
P: 0.942812670644
mAP: 0.792656319873
F: 0.886644477273
旋转， 2000张  3000 nms  iou 0.5
R: 0.825471698113
P: 0.933186075468
mAP: 0.773740173667
F: 0.876030238451
旋转， 2000张  3000 nms  iou 0.6
R: 0.804402515723
P: 0.907360638473
mAP: 0.733757840362
F: 0.852785244812
旋转， 2000张  3000 nms  iou 0.7
R: 0.740566037736
P: 0.837118962891
mAP: 0.624056832155
F: 0.785888023548
"""

# root path
ROOT_PATH = os.path.abspath('/mnt/cirrus/models/R2CNN_FPN_Tensorflow')

# pretrain weights path
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_result'

NET_NAME = 'resnet_v1_101'
VERSION = 'v5'
CLASS_NUM = 1
LEVEL = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
STRIDE = [4, 8, 16, 32, 64]
ANCHOR_SCALES = [1.]
ANCHOR_RATIOS = [1 / 3., 1., 3.0]
SCALE_FACTORS = [10., 10., 5., 5., 5.]
OUTPUT_STRIDE = 16
SHORT_SIDE_LEN = 600 #318 #102 #45 #600
DATASET_NAME = 'building'

BATCH_SIZE = 1
WEIGHT_DECAY = {'resnet_v1_50': 0.0001, 'resnet_v1_101': 0.0001}
EPSILON = 1e-5
MOMENTUM = 0.9
MAX_ITERATION = 100001
GPU_GROUP = "0"
LR = 0.001

# rpn
SHARE_HEAD = False
RPN_NMS_IOU_THRESHOLD = 0.7
MAX_PROPOSAL_NUM = 300
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
IS_FILTER_OUTSIDE_BOXES = True
RPN_TOP_K_NMS = 3000
FEATURE_PYRAMID_MODE = 0  # {0: 'feature_pyramid', 1: 'dense_feature_pyramid'}

# fast rcnn
ROTATE_NMS_USE_GPU = True
FAST_RCNN_MODE = 'build_fast_rcnn1'
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 0.5
FAST_RCNN_NMS_IOU_THRESHOLD = 0.15
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 20
FINAL_SCORE_THRESHOLD = 0.9
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_MINIBATCH_SIZE = 128
FAST_RCNN_POSITIVE_RATE = 0.25
