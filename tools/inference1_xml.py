# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../')

import time, glob
from data.io import image_preprocess
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn
from help_utils.help_utils import *
from help_utils.tools import *
from libs.configs import cfgs
from tools import restore_model
from libs.fast_rcnn import build_fast_rcnn1

#import libs.box_utils.show_box_in_tensor as show_box_in_tensor

from libs.box_utils.coordinate_convert import forward_convert

import cv2
import numpy as np

import sys
sys.path.append('../../')
#from coord_reorder_gt import *
from libs.box_utils.coordinate_convert import forward_convert
from libs.box_utils.coordinate_convert import back_forward_convert

from operator import itemgetter
import itertools
from scipy.spatial import distance as dist

from xml.dom import minidom
import xmltodict
import argparse
from scipy.spatial import distance as dist
import numpy as np
import cv2
from operator import itemgetter
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import itertools


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# START XML STUFF

def XML(imagename, boxCoords, imageShape):
    top = Element('Annotation')
    folder = SubElement(top,'folder')
    folder.text = 'VOC2020'
    filename = SubElement(top,'filename')
    filename.text = imagename + '.jpg'
    path = SubElement(top, 'path')
    path.text= 'prediction_images/'+imagename+ '.jpg'
    for boxCoord in boxCoords:
        objects = SubElement(top, 'object')
        name = SubElement(objects,'name')
        name.text='building'
        pose = SubElement(objects,'pose')
        pose.text='Unknown'
        truncated = SubElement(objects,'truncated')
        truncated.text='0'
        difficult = SubElement(objects,'difficult')
        difficult.text='0'
        secondchild = SubElement(objects,'bndbox')
        grandchild1 = SubElement(secondchild, 'x0')
        grandchild1.text= str(boxCoord[0])
        grandchild2 = SubElement(secondchild, 'y0')
        grandchild2.text = str(boxCoord[1])
        grandchild3 = SubElement(secondchild, 'x1')
        grandchild3.text = str(boxCoord[2])
        grandchild4 = SubElement(secondchild, 'y1')
        grandchild4.text = str(boxCoord[3])
        grandchild5 = SubElement(secondchild, 'x2')
        grandchild5.text = str(boxCoord[4])
        grandchild6 = SubElement(secondchild, 'y2')
        grandchild6.text = str(boxCoord[5])
        grandchild7 = SubElement(secondchild, 'x3')
        grandchild7.text = str(boxCoord[6])
        grandchild8 = SubElement(secondchild, 'y3')
        grandchild8.text = str(boxCoord[7])
    size = SubElement(top,'size')
    width = SubElement(size, 'width')
    width.text = str(imageShape[1])
    height = SubElement(size, 'height')
    height.text = str(imageShape[0])
    depth = SubElement(size, 'depth')
    depth.text = str(3)
    #    print(prettify(top))
    tree = ElementTree.ElementTree(top)
    tree.write("prediction_annotations/"+imagename+"_pred"+".xml")
    return tree
    
    
def order_points(pts):
        print("input pts: ", pts)
        # sort the points based on their x-coordinates
        xSorted = sorted(pts,key=itemgetter(0))
        print("x-sorted pts: ", xSorted)

        # grab the left-most and right-most points from the sorted
        # x-roodinate points

        leftMost = xSorted[:2]
        rightMost = xSorted[2:]
        print("leftMost: ", leftMost, "rightMost: ", rightMost)

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = sorted(leftMost,key=itemgetter(1))
        print("leftMost sorted: ", leftMost)
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        tl = np.asarray(tl)
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        rightMost = np.asarray(rightMost)
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order

        orderedPts = np.array([tl, tr, br, bl], dtype="float32")

        flat_points_list = [item for sublist in orderedPts for item in sublist]
        #print(flat_points_list)
        #flat_points_list = [int(i) for i in flat_points_list]
        print(flat_points_list)

        return flat_points_list


def read_reorder(coordinates):
    if len(coordinates) > 0:
        print("INPUT COORDINATES: ", coordinates)
        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i+n]
        def chunks1(l, n):
            return [l[i:i+n] for i in range(0, len(l), n)]
        flat_points_list = [item for sublist in coordinates for item in sublist]
        if len(flat_points_list) > 4:
            if flat_points_list:
                print("MULTI_BOX")
                flat_points_list = [item for sublist in coordinates for item in sublist]
                flat_points_list_angle = [item for sublist in coordinates for item in sublist]
                print("points_list_angle: ", flat_points_list_angle)
                del flat_points_list[5-1::5]
                print("points_list: ", flat_points_list)
                flat_points_array = np.array(flat_points_list)
                boxes = []
                for rect in coordinates:
                    box = cv2.boxPoints(((rect[1], rect[0]), (rect[3], rect[2]), rect[4]))
                    box = np.reshape(box, [-1, ])
                    boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])
                flat_points_list = [item for sublist in boxes for item in sublist]
                chunked = chunks1(flat_points_list, 8)
                chunked_with_angle = chunks1(flat_points_list_angle, 5)
                print("multi-box chunked: ", chunked)

                ordered_points_list = []
                ordered_points_list_minmax = []
                for chunk in chunked:
                    pairs = list(zip(*[iter(chunk)] * 2))
                    ordered_points_list_chunk = order_points(pairs)
                    ordered_points_list.append(ordered_points_list_chunk)
                    ymin1 = min(ordered_points_list_chunk[5], ordered_points_list_chunk[7])
                    xmin1 = min(ordered_points_list_chunk[0], ordered_points_list_chunk[6])
                    ymax1 = max(ordered_points_list_chunk[1], ordered_points_list_chunk[3])
                    xmax1 = max(ordered_points_list_chunk[2], ordered_points_list_chunk[4])
                    ordered_points_list_chunk_minmax = [ymin1, xmin1, ymax1, xmax1]
                    ordered_points_list_minmax.append(ordered_points_list_chunk_minmax)


                boxes_ordered = []
                for rect in ordered_points_list:
                    box = np.int0(rect)
                    box = box.reshape([4, 2])
                    rect1 = cv2.minAreaRect(box)

                    x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
                    boxes_ordered.append([y, x, h, w, theta])

                for chunk1, chunk0, chunk_mm in zip(ordered_points_list, chunked_with_angle, ordered_points_list_minmax):
                    print("ordered_points_list chunk: ", chunk1)
                    print("angle: ", chunk0[-1])
                    angle = float(chunk0[-1])
                    #chunk1 = chunk1.append(angle)
                    chunk_mm = chunk_mm.append(angle)
            else:
                print("multi-box")
                flat_points_array = np.array(flat_points_list)
                chunked = chunks1(flat_points_list, 4)
                print("multi-box chunked: ", chunked)

                ordered_points_list = []
                chunk_expand_list = []
                for chunk in chunked:
                    ymin, xmin, ymax, xmax = chunk[0], chunk[1], chunk[2], chunk[3]
                    print("chunk ymin, xmin, ymax, xmax: ", ymin, xmin, ymax, xmax)
                    chunk_expand = [xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin]
                    chunk_expand_list.append(chunk_expand)
                    pairs = list(zip(*[iter(chunk_expand)] * 2))
                    ordered_points_list_chunk = order_points(pairs)
                    ordered_points_list.append(ordered_points_list_chunk)
            print("original points list: ", coordinates)
            print("ordered_points_list: ", ordered_points_list)
            print("ordered_points_list_minmax: ", ordered_points_list_minmax)
        else:
            print("single box")
            ymin, xmin, ymax, xmax = flat_points_list[0], flat_points_list[1], flat_points_list[2], flat_points_list[3]
            chunk_expand = list(xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin)
            pairs = list(zip(*[iter(chunk_expand)] * 2))
            ordered_points_list = []
            ordered_points_list = [order_points(pairs)]
            print("original points list: ", coordinates)
            print("expanded points list: ", chunk_expand)
            print("ordered_points_list: ", ordered_points_list)
        return np.array(ordered_points_list, dtype=np.float32)
    else:
        return np.array([], dtype=np.float32)


def writePredictionsXML(filename, coords, img_shape):
    filename_split = os.path.splitext(filename)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    print("file: ", basename)
    #tree = ElementTree.ElementTree(file=filename)
    #root = tree.getroot()
    coordinates = read_reorder(coords)
    xmlOut = XML(basename, coordinates, img_shape)
    return xmlOut

# STOP XML STUFF

def get_imgs():
    mkdir(cfgs.INFERENCE_IMAGE_PATH)
    root_dir = cfgs.INFERENCE_IMAGE_PATH
    print("root_dir: ", root_dir)
    img_name_list = os.listdir(root_dir)
    if len(img_name_list) == 0:
        assert 'no test image in {}!'.format(cfgs.INFERENCE_IMAGE_PATH)
    img_list = [cv2.imread(os.path.join(root_dir, img_name))
                for img_name in img_name_list]
    return img_list, img_name_list


def inference():
    with tf.Graph().as_default():

        img_plac = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

        img_tensor = tf.cast(img_plac, tf.float32) - tf.constant([103.939, 116.779, 123.68])
        img_batch = image_preprocess.short_side_resize_for_inference_data(img_tensor,
                                                                          target_shortside_len=cfgs.SHORT_SIDE_LEN)
        img_name_plac = tf.placeholder(tf.string)

        # ***********************************************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                          inputs=img_batch,
                                          num_classes=None,
                                          is_training=True,
                                          output_stride=None,
                                          global_pool=False,
                                          spatial_squeeze=False)
        # ***********************************************************************************************
        # *                                            RPN                                              *
        # ***********************************************************************************************
        rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                            inputs=img_batch,
                            gtboxes_and_label=None,
                            is_training=False,
                            share_head=cfgs.SHARE_HEAD,
                            share_net=share_net,
                            stride=cfgs.STRIDE,
                            anchor_ratios=cfgs.ANCHOR_RATIOS,
                            anchor_scales=cfgs.ANCHOR_SCALES,
                            scale_factors=cfgs.SCALE_FACTORS,
                            base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                            level=cfgs.LEVEL,
                            top_k_nms=cfgs.RPN_TOP_K_NMS,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=False,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

        # rpn predict proposals
        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************
        fast_rcnn = build_fast_rcnn1.FastRCNN(feature_pyramid=rpn.feature_pyramid,
                                              rpn_proposals_boxes=rpn_proposals_boxes,
                                              rpn_proposals_scores=rpn_proposals_scores,
                                              img_shape=tf.shape(img_batch),
                                              #filename=img_name_plac,
                                              roi_size=cfgs.ROI_SIZE,
                                              roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                              scale_factors=cfgs.SCALE_FACTORS,
                                              gtboxes_and_label=None,
                                              gtboxes_and_label_minAreaRectangle=None,
                                              fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                              fast_rcnn_maximum_boxes_per_img=100,
                                              fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                              show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                              # show detections which score >= 0.6
                                              num_classes=cfgs.CLASS_NUM,
                                              fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                              fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                              fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                              # iou>0.5 is positive, iou<0.5 is negative
                                              use_dropout=cfgs.USE_DROPOUT,
                                              weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                              is_training=False,
                                              level=cfgs.LEVEL)

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category, \
        fast_rcnn_decode_boxes_rotate_original, fast_rcnn_decode_boxes_rotate, fast_rcnn_decode_boxes_rotate_reorder, fast_rcnn_score_rotate, num_of_objects_rotate, detection_category_rotate = \
            fast_rcnn.fast_rcnn_predict()

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = restore_model.get_restorer()

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            
            imgs, img_names = get_imgs()
            #print("img_names: ", img_names)
            for i, (img, img_name) in enumerate(zip(imgs, img_names)):
            #for i, img in enumerate(imgs):

                start = time.time()

                _img_batch, _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category, \
                _fast_rcnn_decode_boxes_rotate,  _fast_rcnn_decode_boxes_rotate_reorder, _fast_rcnn_score_rotate, _detection_category_rotate = \
                    sess.run([img_batch, fast_rcnn_decode_boxes, fast_rcnn_score, detection_category,
                              fast_rcnn_decode_boxes_rotate, fast_rcnn_decode_boxes_rotate_reorder, fast_rcnn_score_rotate, detection_category_rotate],
                             feed_dict={img_plac: img}) #, img_name_plac: img_name})
                end = time.time()

                img_np = np.squeeze(_img_batch, axis=0)

                print("_fast_rcnn_decode_boxes: ", _fast_rcnn_decode_boxes)

                img_horizontal_np = draw_box_cv(img_np,
                                                boxes=_fast_rcnn_decode_boxes,
                                                labels=_detection_category,
                                                scores=_fast_rcnn_score)

                print("_fast_rcnn_decode_boxes_rotate: ", _fast_rcnn_decode_boxes_rotate)

                img_rotate_np = draw_rotate_box_cv(img_np,
                                                   boxes=_fast_rcnn_decode_boxes_rotate,
                                                   labels=_detection_category_rotate,
                                                   scores=_fast_rcnn_score_rotate)

                xml_rotate_np = writePredictionsXML(img_name, _fast_rcnn_decode_boxes_rotate, img.shape)

                mkdir(cfgs.INFERENCE_SAVE_PATH)
                print("save_dir: ", cfgs.INFERENCE_SAVE_PATH)
                cv2.imwrite(cfgs.INFERENCE_SAVE_PATH + '/{}_horizontal_fpn.jpg'.format(img_names[i]), img_horizontal_np)
                cv2.imwrite(cfgs.INFERENCE_SAVE_PATH + '/{}_rotate_fpn.jpg'.format(img_names[i]), img_rotate_np)
                view_bar('{} cost {}s'.format(img_names[i], (end - start)), i + 1, len(imgs))
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    inference()
                 
