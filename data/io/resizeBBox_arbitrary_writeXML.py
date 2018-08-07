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

from PIL import Image

import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--convertPath', '-p', help="path for old annotations")
parser.add_argument('--imgPath', '-i', help="path for old annotations")

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
    tree.write("unit_test_GT_annotations_resized/"+imagename+".xml")
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

def cvResizeBBs(bb, imgShapeXML, imgShape):
    # Note: flipped comparing to your original code!

    y_ = int(imgShapeXML[0])
    x_ = int(imgShapeXML[1])

    targetSizeX = int(imgShape[1])
    targetSizeY = int(imgShape[0])
    #x_scale = targetSizeX / x_
    #y_scale = targetSizeY / y_
    
    x_scale = x_ / targetSizeX
    y_scale = y_ / targetSizeY
    
    print(x_scale, y_scale)

    # original frame as named values

    x0 = bb[0]
    x1 = bb[2]
    x2 = bb[4]
    x3 = bb[6]


    y0 = bb[1]
    y1 = bb[3]
    y2 = bb[5]
    y3 = bb[7]


    x0 = int(np.round(x0 * x_scale))
    y0 = int(np.round(y0 * y_scale))
    x1 = int(np.round(x1 * x_scale))
    y1 = int(np.round(y1 * y_scale))
    x2 = int(np.round(x2 * x_scale))
    y2 = int(np.round(y2 * y_scale))
    x3 = int(np.round(x3 * x_scale))
    y3 = int(np.round(y3 * y_scale))

    print("original bbox: ", bb)
    print("resized bbox: ", [x0,y0,x1,y1,x2,y2,x3,y3])
    return np.array([x0,y0,x1,y1,x2,y2,x3,y3])

def read_reorder(filename, img):
    tree = ElementTree.ElementTree(file=filename)
    root = tree.getroot()
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]
    points_list = []
    for name in root.findall(".//bndbox/"):
        points = [name.text]
        #print("points: ", points)
        points_list.append(points)
    flat_points_list = [item for sublist in points_list for item in sublist]
    print("points_list: ", flat_points_list)
    pairs = list(zip(*[iter(flat_points_list)] * 2))
    chunked = list(chunks(pairs, 4))
    print("chunked: ", chunked)
    print("pairs: ", pairs)
    ordered_points_list = []
    for chunk in chunked:
        ordered_points_list_chunk = order_points(chunk)
        ordered_points_list.append(ordered_points_list_chunk)
    img_height = tree.find('.//height').text
    img_width = tree.find('.//width').text
    img_depth = tree.find('.//depth').text
    print("image shape: ", img_height, img_width, img_depth)
    img_shapeXML = [img_height, img_width, img_depth]
    img = np.array(Image.open(img))
    img_shape = img.shape
    coordinates = chunked

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
                print("multi-box")
                flat_points_array = np.array(flat_points_list)
                chunked = chunks1(flat_points_list, 4)
                print("multi-box chunked: ", chunked)

                ordered_points_list = []
                chunk_expand_list = []
                for chunk in chunked:
                    #ymin, xmin, ymax, xmax = chunk[0], chunk[1], chunk[2], chunk[3]
                    #print("chunk ymin, xmin, ymax, xmax: ", ymin, xmin, ymax, xmax)
                    #chunk_expand = [xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin]
                    #chunk_expand_list.append(chunk_expand)
                    #pairs = list(zip(*[iter(chunk_expand)] * 2))
                    #ordered_points_list_chunk = order_points(pairs)
                    #ordered_points_list_chunk = order_points(chunk_expand)
                    ordered_points_list_chunk = order_points(chunk)
                    ordered_points_list.append(ordered_points_list_chunk)
            print("original points list: ", coordinates)
            print("ordered_points_list: ", ordered_points_list)
            #print("ordered_points_list_minmax: ", ordered_points_list_minmax)
            bbox_resized_list = []
            for bbox in ordered_points_list:
                bbox_resize = cvResizeBBs(bbox, img_shape, img_shapeXML)
                bbox_resized_list.append(bbox_resize)
        else:
            print("single box")
            #ymin, xmin, ymax, xmax = flat_points_list[0], flat_points_list[1], flat_points_list[2], flat_points_list[3]
            #chunk_expand = list(xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin)
            #pairs = list(zip(*[iter(chunk_expand)] * 2))
            ordered_points_list = []
            #ordered_points_list = [order_points(pairs)]
            ordered_points_list = [order_points(flat_points_list)]
            print("original points list: ", coordinates)
            #print("expanded points list: ", chunk_expand)
            print("ordered_points_list: ", ordered_points_list)
            #for bbox in ordered_points_list:
            ordered_points_list = [item for sublist in ordered_points_list for item in sublist]
            bbox_resized_list = cvResizeBBs(ordered_points_list, img_shape, img_shapeXML)
            bbox_resized_list = [bbox_resized_list]
        #return np.array(ordered_points_list, dtype=np.float32), img_shape
        #img_shape = [512,512,3]
        return np.array(bbox_resized_list, dtype=np.float32), img_shape
    else:
        return np.array([], dtype=np.float32), img_shape
        
       
def writePredictionsXML(filename, img):
    filename_split = os.path.splitext(filename)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    print("file: ", basename)
    #tree = ElementTree.ElementTree(file=filename)
    #root = tree.getroot()
    coordinates, img_shape = read_reorder(filename, img)
    xmlOut = XML(basename, coordinates, img_shape)
    return xmlOut

# STOP XML STUFF

def main():
    for infile in glob.glob(args.convertPath+"*.xml"):
        print("infile: ", infile)
        filename_split = os.path.splitext(infile)
        filename_zero, fileext = filename_split
        basename = os.path.basename(filename_zero)
        print("basename: ", basename)
        XMLout = writePredictionsXML(args.convertPath+basename+".xml", args.imgPath+basename+".jpg")
    print("done")

main()
