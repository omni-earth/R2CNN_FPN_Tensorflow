import numpy as np
import xml.etree.ElementTree as xmlParser
import xml.etree.cElementTree as ET
import os, sys, glob
from xml.dom import minidom
import xmltodict
import argparse

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
    print("chunked: ", list(chunks(flat_points_list, 8)))
    chunked = list(chunks(flat_points_list, 8))
    img_height = tree.find('.//height').text
    img_width = tree.find('.//width').text
    img_shape = np.array(img_height, img_width)
    filename = basename
    return filename, chunked, img_shape
    

def run(f):
    imagename, boxCoords, imageShape = getInfo(f)
    XML(imagename, boxCoords, imageShape)
    return
    
for f in glob.glob('./Annotations/'+'*.xml'):
    parse(f)

 
