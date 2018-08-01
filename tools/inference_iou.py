# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from libs.box_utils.coordinate_convert import *
from libs.box_utils.rbbox_overlaps import rbbx_overlaps
from libs.box_utils.iou_cpu import get_iou_matrix
import argparse
import math
import os, glob

import xml.etree.cElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--gtPath', '-p', help="path for rewritten annotations")
parser.add_argument('--predPath', '-i', help="path for images")
parser.add_argument('--iouPath', '-a', help="path for annotations")


args = parser.parse_args()


def iou_rotate_calculate(boxes1, boxes2, use_gpu=True, gpu_id=0):
    '''
    :param boxes_list1:[N, 8] tensor
    :param boxes_list2: [M, 8] tensor
    :return:
    '''

    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)
    if use_gpu:

        iou_matrix = tf.py_func(rbbx_overlaps,
                                inp=[boxes1, boxes2, gpu_id],
                                Tout=tf.float32)
    else:
        iou_matrix = tf.py_func(get_iou_matrix, inp=[boxes1, boxes2],
                                Tout=tf.float32)

    iou_matrix = tf.reshape(iou_matrix, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return iou_matrix


def iou_rotate_calculate1(boxes1, boxes2, use_gpu=True, gpu_id=0):

    # start = time.time()
    if use_gpu:
        ious = rbbx_overlaps(boxes1, boxes2, gpu_id)
    else:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        ious = []
        for i, box1 in enumerate(boxes1):
            temp_ious = []
            r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
            for j, box2 in enumerate(boxes2):
                r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                    temp_ious.append(inter)
                else:
                    temp_ious.append(0.0)
            ious.append(temp_ious)

    # print('{}s'.format(time.time() - start))

    return np.array(ious, dtype=np.float32)

def back_forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)] 
    :param with_label: default True
    :return: format [y_c, x_c, h, w, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([y, x, h, w, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([y, x, h, w, theta])

    return np.array(boxes, dtype=np.float32)
    
def getInfo(f):
    filename_split = os.path.splitext(f)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    print("file: ", basename)
    tree = ET.ElementTree(file=f)
    root = tree.getroot()
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]
    points_list = []
    for name in root.findall(".//bndbox/"):
        points = [name.text]
        #points = [int(point) for point in points]
        #points = map(int, points)
        #print("points: ", points)
        points_list.append(points)
    flat_points_list = [item for sublist in points_list for item in sublist]
    print("points_list: ", flat_points_list)
    pairs = list(zip(*[iter(flat_points_list)] * 2))
    chunked = list(chunks(pairs, 4))
    return chunked

def main(gt, pred):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '13'
    #boxes1 = np.array([[50, 50, 100, 300, 0],
    #                  [60, 60, 100, 200, 0]], np.float32)

    #boxes2 = np.array([[50, 50, 100, 300, -45.],
    #                   [200, 200, 100, 200, 0.]], np.float32)

    boxes1 = getInfo(gt)
    boxes2 = getInfo(pred)
    
    boxes1 = getInfo(gt)
    boxes2 = getInfo(pred)


    with tf.Graph().as_default():
        with tf.name_scope('get_boxes'):
            boxes1 = back_forward_convert(boxes1, with_label=False)
            boxes2 = back_forward_convert(boxes2, with_label=False)
            ious = iou_rotate_calculate1(boxes1, boxes2, use_gpu=False)

    init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())

    start = time.time()
    #myGraph = tf.Graph()
    with tf.Session() as sess:
        #ious = iou_rotate_calculate1(boxes1, boxes2, use_gpu=False)
        #print(sess.run(ious))
        print(sess.run(init_op))
        print('{}s'.format(time.time() - start))
        return ious

   #if len(boxes1) > 1:
   #    for box in boxes1:
   #        start = time.time()
   #        with tf.Session() as sess:
   #            ious = iou_rotate_calculate1(box, boxes2, use_gpu=False)
   #            print(sess.run(ious))
   #            print('{}s'.format(time.time() - start))


    # start = time.time()
    # for _ in range(10):
    #     ious = rbbox_overlaps.rbbx_overlaps(boxes1, boxes2)
    # print('{}s'.format(time.time() - start))
    # print(ious)

    # print(ovr)

for gt, pred in zip(glob.glob(args.gtPath+"*.xml"), glob.glob(args.predPath+"*.xml")):
    filename_split = os.path.splitext(gt)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    ious = main(args.gtPath+basename+".xml", args.predPath+basename+".xml")
    print("ious: ", ious)

