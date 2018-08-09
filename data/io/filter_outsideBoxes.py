import numpy as np
import xml.etree.ElementTree as xmlParser
import xml.etree.cElementTree as ET
import os, sys, glob, shutil
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

import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import random
import numpy as np

from PIL import Image

import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import random
import numpy as np

from PIL import Image, ImageChops

import numpy
import scipy.ndimage
import scipy.interpolate
import pdb
from scipy import ndimage as nd
from scipy.stats import itemfreq

from skimage.segmentation import clear_border


parser = argparse.ArgumentParser()
parser.add_argument('--outPath', '-p', help="path for rewritten annotations")
parser.add_argument('--imPath', '-i', help="path for images")
parser.add_argument('--annPath', '-a', help="path for annotations")


args = parser.parse_args()

def XML(imagename, boxCoords, imageShape, d):
    top = Element('Annotation')
    folder = SubElement(top,'folder')
    folder.text = 'VOC2020'
    filename = SubElement(top,'filename')
    filename.text = imagename + '.jpg'
    path = SubElement(top, 'path')
    path.text= args.imPath+imagename+ '.jpg'
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
    xmlName = args.outPath+imagename+'_rotate'+str(d)
    tree = ET.ElementTree(top)
    tree.write(args.outPath+imagename+'.xml')
    return tree

def order_points(pts):
        # sort the points based on their x-coordinates
        xSorted = sorted(pts,key=itemgetter(0))
        print(xSorted)

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
        print(tl.shape)
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        rightMost = np.asarray(rightMost)
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order

        orderedPts = np.array([tl, tr, br, bl], dtype="float32")

        flat_points_list = [item for sublist in orderedPts for item in sublist]
        print(flat_points_list)
        flat_points_list = [int(i) for i in flat_points_list]
        print(flat_points_list)
        return flat_points_list
      

def getInfo(f, d):
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
        #print("points: ", points)
        points_list.append(points)
    flat_points_list = [int(item) for sublist in points_list for item in sublist]
    print("points_list: ", flat_points_list)
    img = args.imPath+basename+'.jpg'
    img_height = tree.find('.//height').text
    img_width = tree.find('.//width').text
    img_depth = tree.find('.//depth').text
    print("image shape: ", img_height, img_width, img_depth)
    img_shapeXML = [img_height, img_width, img_depth]
    img = np.array(Image.open(img))
    img_shape = img.shape
    pairs = list(zip(*[iter(flat_points_list)] * 2))
    chunked = list(chunks(pairs, 4))
    print("chunked: ", chunked)
    print("pairs: ", pairs)
    ordered_points_list = []
    print("original points list: ", chunked)
    print("ordered_points_list: ", ordered_points_list)

    filename_aug = basename

    xSorted = sorted(pairs,key=itemgetter(0))
    xSorted_extract = [i[0] for i in xSorted]
    xmin = int(min(xSorted_extract))
    xmax = int(max(xSorted_extract))
    ySorted = sorted(pairs,key=itemgetter(1))
    ySorted_extract = [i[1] for i in ySorted]
    ymin = int(min(ySorted_extract))
    ymax = int(min(ySorted_extract))

    img_xmin = 0
    img_xmax = img_shape[0]
    img_ymin = 0
    img_ymax = img_shape[1]
    for chunk in chunked:
        chunk = [int(item) for sublist in chunk for item in sublist]
        x0,y0,x1,y1,x2,y2,x3,y3 = chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]
        ymin, xmin, ymax, xmax = min(y0,y1,y2,y3), min(x0,x1,x2,x3), max(y0,y1,y2,y3), max(x0,x1,x2,x3)

        print("xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)
        print("img_xmin, img_xmax, img_ymin, img_ymax: ", img_xmin, img_xmax, img_ymin, img_ymax)

        if xmin < img_xmin:
            out_of_bounds.append(filename_aug)
            print("box off frame for file: ", filename_aug)
        elif xmax > img_xmax:
            out_of_bounds.append(filename_aug)
            print("box off frame for file: ", filename_aug)
        elif ymin < img_ymin:
            out_of_bounds.append(filename_aug)
            print("box off frame for file: ", filename_aug)
        elif ymax > img_ymax:
            out_of_bounds.append(filename_aug)
            print("box off frame for file: ", filename_aug)
        else:
            print("box within frame")

    return filename_aug, ordered_points_list, img_shape
    
    
def run(f):
    for d in [x for x in range(370) if x % 10 == 0][1:]:
        imagename, boxCoords, imageShape = getInfo(f, d)
        #XML(imagename, boxCoords, imageShape, d)
    return

out_of_bounds = []

for f in glob.glob(args.annPath+'*.xml'):
    run(f)

out_of_bounds_outfile = open('out_of_bounds.txt', 'w')
for item in out_of_bounds:
    out_of_bounds_outfile.write("%s\n" % item)

for f in out_of_bounds:
    if os.path.isfile(args.annPath+f+".xml"):
        shutil.move(args.annPath+f+".xml", args.annPath+"inside/"+f+".xml")


    
