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
from PIL import Image
import re

parser = argparse.ArgumentParser()
parser.add_argument('--outpath', '-p', help="path for rewritten annotations")

args = parser.parse_args()

def XML(imagename, boxCoords, imageShape, names):
    top = Element('Annotation')
    folder = SubElement(top,'folder')
    folder.text = 'VOC2020'
    filename = SubElement(top,'filename')
    filename.text = imagename + '.jpg'
    path = SubElement(top, 'path')
    path.text= 'JPEGImages/'+imagename+ '.jpg'
    for boxCoord, objname in zip(boxCoords, names):
        print("boxCoord: ", boxCoord)
        objname = str(objname)
        objname = objname.strip("[]")
        objname = objname.replace("'", "")
        print("name: ", objname)
        objects = SubElement(top, 'object')
        name = SubElement(objects,'name')
        name.text=str(objname)
        pose = SubElement(objects,'pose')
        pose.text='Unknown'
        truncated = SubElement(objects,'truncated')
        truncated.text='0'
        difficult = SubElement(objects,'difficult')
        difficult.text='0'
        secondchild = SubElement(objects,'bndbox')
        grandchild1 = SubElement(secondchild, 'x0')
        grandchild1.text= str(boxCoord[0][0])
        grandchild2 = SubElement(secondchild, 'y0')
        grandchild2.text = str(boxCoord[0][1])
        grandchild3 = SubElement(secondchild, 'x1')
        grandchild3.text = str(boxCoord[0][2])
        grandchild4 = SubElement(secondchild, 'y1')
        grandchild4.text = str(boxCoord[0][3])
        grandchild5 = SubElement(secondchild, 'x2')
        grandchild5.text = str(boxCoord[0][4])
        grandchild6 = SubElement(secondchild, 'y2')
        grandchild6.text = str(boxCoord[0][5])
        grandchild7 = SubElement(secondchild, 'x3')
        grandchild7.text = str(boxCoord[0][6])
        grandchild8 = SubElement(secondchild, 'y3')
        grandchild8.text = str(boxCoord[0][7])
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


def getInfo(f):
    filename_split = os.path.splitext(f)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    print("file: ", basename)
    pngname = './images/'+basename+'.png'
    png = Image.open(pngname)
    png.save('./images/'+basename+'.jpg', "JPEG")
    imagename = './images/'+basename+'.jpg'
    image = np.array(Image.open(imagename))
    lines = []
    with open(f, 'r') as f:
        f = f.read()
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip('\r')
            lines.append(line)
    lines  = ''.join(map(str, lines))
    lines = [i for i in lines.split()]
    print("lines: ", lines)
    print("file header: ", lines[0:2])
    print("sample set of boxes: ", lines[3:10])
    def chunks(l, n):
            for i in range(0, len(l), n):
            yield l[i:i+n]
    points_list = []
    names = []
    lines = list(chunks(lines, 9))
    for bbox in lines:
        points = bbox[0:8]
        print("points: ", points)
        points_list.append(points)
        name = bbox[-1]
        print("name: ", name)
        names.append(name)
    del points_list[-1]
    del names[-1]
    chunked_points = list(chunks(points_list, 8))
    chunked_names = list(chunks(names, 1))
    img_shape = image.shape
    print("image shape: ", img_shape)
    filename = basename
    print("chunked points list: ", chunked_points)
    print("chunked names list: ", chunked_names)
    return filename, chunked_points, img_shape, chunked_names

def run(f):
    imagename, boxCoords, imageShape, names = getInfo(f)
    XML(imagename, boxCoords, imageShape, names)
    return

# delete first two lines of all label files first using BASH line: ```sed -i.bak -n '3,$p' P*```
for f in glob.glob('./labels/'+'*.txt'):
    run(f)
    
