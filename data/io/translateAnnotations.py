# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../')

import time
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

import sys, glob
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

import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--convertPath', '-p', help="path for old annotations")

args = parser.parse_args()

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
    tree.write("danPredictions_reformatted/"+imagename+".xml")
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
        flat_points_list = [int(i) for i in flat_points_list]
        print(flat_points_list)

        return flat_points_list
        
def readOld(infile):
    if infile:
        print("infile: ", infile)
        filename_split = os.path.splitext(infile)
        filename_zero, fileext = filename_split
        basename = os.path.basename(filename_zero)
        with open(infile) as f:
            reader = csv.reader(f)
            next(reader) # skip header
            data = [r for r in reader]
            print("csv content: ", data)
            xmin = float(data[0][2]) * 512
            ymin = float(data[0][3]) * 512
            xmax = float(data[0][4]) * 512
            ymax = float(data[0][5]) * 512

            print("xmin, ymin, xmax, ymax: ", xmin, ymin, xmax, ymax)

            flat_points_list = [xmin, ymin, xmax, ymax]
            print("points_list: ", flat_points_list)
            flat_points_array = np.array(flat_points_list)
            chunk_expand_list = []
            ordered_points_list = []
            if len(flat_points_array) > 4:
                for rect in flat_points_array:
                #box = cv2.boxPoints(((rect[1], rect[0]), (rect[3], rect[2]), rect[4]))
                #box = np.reshape(box, [-1, ])
                #boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])
                    ymin, xmin, ymax, xmax = rect[0], rect[1], rect[2], rect[3]
                    chunk_expand = [xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin]
                    chunk_expand_list.append(chunk_expand)
                    pairs = list(zip(*[iter(chunk_expand)] * 2))
                    ordered_points_list_chunk = order_points(pairs)
                    ordered_points_list.append(ordered_points_list_chunk)
            else:
                ymin, xmin, ymax, xmax = int(flat_points_array[0]), int(flat_points_array[1]), int(flat_points_array[2]), int(flat_points_array[3])
                chunk_expand = [xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin]
                chunk_expand_list.append(chunk_expand)
                pairs = list(zip(*[iter(chunk_expand)] * 2))
                ordered_points_list_chunk = order_points(pairs)
                ordered_points_list.append(ordered_points_list_chunk)
            print("original points list: ", chunk_expand_list)
            print("ordered_points_list: ", ordered_points_list)
            flat_points_list = [item for sublist in ordered_points_list for item in sublist]
            def chunks(l, n):
                for i in range(0, len(l), n):
                    yield l[i:i+n]
            chunked = chunks(flat_points_list, 8)
            #chunked_with_angle = chunks1(flat_points_list_angle, 5)
            #print("multi-box chunked: ", chunked)
            img_shape = np.array([512,512,3])
            #flat_points_list_non_nested = flat_points_list[0]
            #flat_points_list_non_nested = [int(i) for i in flat_points_list_non_nested]
            return basename, ordered_points_list, img_shape

def main():
    for infile in glob.glob(args.convertPath+"*.csv"):
        basename, flat_points_list, img_shape = readOld(infile)
        XML(basename, flat_points_list, img_shape)
    print("done")

main()
      
