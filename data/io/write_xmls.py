import numpy as np
import xml.etree.ElementTree as xmlParser
import xml.etree.cElementTree as ET
import os, sys, glob
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

parser = argparse.ArgumentParser()
parser.add_argument('--outpath', '-p', help="path for rewritten annotations")

args = parser.parse_args()


def XML(imagename, boxCoords, imageShape):
    top = Element('Annotation')
    folder = SubElement(top,'folder')
    folder.text = 'VOC2020'
    filename = SubElement(top,'filename')
    filename.text = imagename + '.jpg'
    path = SubElement(top, 'path')
    path.text= 'JPEGImages/'+imagename+ '.jpg'
    
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
    tree = ET.ElementTree(top)
    tree.write(args.outpath+imagename+".xml")
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

def getInfo(f):
    filename_split = os.path.splitext(f)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
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
    flat_points_list = [item for sublist in points_list for item in sublist]
    print("points_list: ", flat_points_list)
    chunked = list(chunks(flat_points_list, 8))
    print("chunked: ", list(chunks(flat_points_list, 8)))
    pairs = list(zip(*[iter(chunked)] * 2))
    ordered_points_list = order_points(pairs)
    img_height = tree.find('.//height').text
    img_width = tree.find('.//width').text
    img_shape = np.array(img_height, img_width)
    filename = basename
    return filename, ordered_points_list, img_shape
    

def run(f):
    imagename, boxCoords, imageShape = getInfo(f)
    XML(imagename, boxCoords, imageShape)
    return
    
for f in glob.glob('./Annotations/'+'*.xml'):
    parse(f)

 
