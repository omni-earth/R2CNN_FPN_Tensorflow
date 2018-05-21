# Intended to change counter-clockwise coordinates from top-right to clockwise coordinates from top-left

import numpy as np
import xml.etree.ElementTree as xmlParser
import xml.etree.cElementTree as ET
import os, sys, glob
from xml.dom import minidom
import xmltodict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outpath', '-p', help="path for rewritten annotations")

args = parser.parse_args()

def parse(f):
    tree = ET.ElementTree(file=f)
    root = tree.getroot()
    filename_split = os.path.splitext(f)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    outpath = args.outpath
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
    #xmlDoc = xmlParser.parse(f)
    #rootElement = xmlDoc.getroot()
    for name in root.findall(".//bndbox/x0"):
        x0 = name.text
        x0_cp = name.text
        for name1 in root.findall(".//bndbox/x1"):
            x1 = name1.text
            x1_cp = name1.text
            x0 = x1
            x1 = x0_cp
            print("old x0 and x1: ", x0_cp, x1_cp, "new x0 and x1: ", x0, x1)
            tree.find('.//bndbox/x0').text = x1
            tree.find('.//bndbox/x1').text = x0
            print("xml x0: ", tree.find('.//bndbox/x0').text)
            print("xml x1: ", tree.find('.//bndbox/x1').text)
    for name in root.findall(".//bndbox/y0"):
        y0 = name.text
        y0_cp = name.text
        for name1 in root.findall(".//bndbox/y1"):
            y1 = name1.text
            y1_cp = name1.text
            y0 = y1
            y1 = y0_cp
            print("old y0 and y1: ", y0_cp, y1_cp, "new y0 and y1: ", y0, y1)
            tree.find('.//bndbox/y0').text = y1
            tree.find('.//bndbox/y1').text = y0
            print("xml y0: ", tree.find('.//bndbox/y0').text)
            print("xml y1: ", tree.find('.//bndbox/y1').text)
    for name in root.findall(".//bndbox/x2"):
        x2 = name.text
        x2_cp = name.text
        for name1 in root.findall(".//bndbox/x3"):
            x3 = name1.text
            x3_cp = name1.text
            x2 = x3
            x3 = x2_cp
            print("old x2 and x3: ", x2_cp, x3_cp, "new x2 and x3: ", x2, x3)
            tree.find('.//bndbox/x2').text = x3
            tree.find('.//bndbox/x3').text = x2
            print("xml x2: ", tree.find('.//bndbox/x2').text)
            print("xml x3: ", tree.find('.//bndbox/x3').text)
    for name in root.findall(".//bndbox/y2"):
        y2 = name.text
        y2_cp = name.text
        for name1 in root.findall(".//bndbox/y3"):
            y3 = name1.text
            y3_cp = name1.text
            y2 = y3
            y3 = y2_cp
            print("old y2 and y3: ", y2_cp, y3_cp, "new y2 and y3: ", y2, y3)
            tree.find('.//bndbox/y2').text = y3
            tree.find('.//bndbox/y3').text = y2
            print("xml y2: ", tree.find('.//bndbox/y2').text)
            print("xml y3: ", tree.find('.//bndbox/y3').text)
    tree.write(outpath+basename+'.xml')
    return chunked

for f in glob.glob('./Annotations/'+'*.xml'):
    parse(f)
