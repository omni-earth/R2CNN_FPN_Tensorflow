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

import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import random
import numpy as np

from PIL import Image

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
    path.text= 'JPEGImages_augment/'+imagename+ '.jpg'
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
    tree.write(args.outpath+imagename+"_rotate"+str(d)+"crop75"+".xml")
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

def augment_img(img, d):
    filename_split = os.path.splitext(f)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    img = np.array(Image.open(img))
    seq = iaa.Sequential([iaa.Affine(rotate=d), iaa.CropAndPad(px=(-75, 0))]) # rotate by exactly d deg
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_images([img])[0]
    #image_before = keypoints.draw_on_image(image, size=7)
    #image_after = keypoints_aug.draw_on_image(image_aug, size=7)
    misc.imsave('JPEGImages_augment/'+basename+'_rotate'+str(d)+'crop75'+'.jpg', image_aug)
    return image_aug

def augment_keypoints(keypoints, img, d):
    ia.seed(1)
    seq = iaa.Sequential([iaa.Affine(rotate=int(d)), iaa.CropAndPad(px=(-75, 0))]) # rotate by exactly d degrees
    seq_det = seq.to_deterministic()
    kpts = ia.KeypointsOnImage([
        ia.Keypoint(x=keypoints[0], y=keypoints[1]),
        ia.Keypoint(x=keypoints[2], y=keypoints[3]),
        ia.Keypoint(x=keypoints[4], y=keypoints[5]),
        ia.Keypoint(x=keypoints[6], y=keypoints[7])
    ], shape=img.shape)
    #keypoints_list.append(kpts)
    keypoints_aug = seq_det.augment_keypoints([kpts])[0]
    for i in range(len(kpts.keypoints)):
        before = kpts.keypoints[i]
        after = keypoints_aug.keypoints[i]
        #print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (i, before.x, before.y, after.x, after.y))
    print("keypoints_aug: ", keypoints_aug)
    new_coords = []
    for kp_idx, keypoint in enumerate(keypoints_aug.keypoints):
        keypoint_old = kpts.keypoints[kp_idx]
        x_old, y_old = keypoint_old.x, keypoint_old.y
        x_new, y_new = keypoint.x, keypoint.y
        #print("[Keypoints for image #%s] before aug: x=%s y=%s | after aug: x=%s y=%s" % (img, x_old, y_old, x_new, y_new))
        new_pair = (int(x_new), int(y_new))
        new_coords.append(new_pair)
    print("new_coords: ", new_coords)
    return new_coords
    
def augment_keypoints_frame(keypoints, img, d):
    ia.seed(1)
    seq = iaa.Sequential([iaa.Affine(rotate=int(d))]) # rotate by exactly d degrees
    seq_det = seq.to_deterministic()
    keypoints_aug = seq_det.augment_keypoints([keypoints])[0]
    for i in range(len(keypoints.keypoints)):
        before = keypoints.keypoints[i]
        after = keypoints_aug.keypoints[i]
        #print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (i, before.x, before.y, after.x, after.y))
    print("keypoints_aug: ", keypoints_aug)
    #bbs_aug = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=keypoints_aug[0], x2=keypoints_aug[2], y1=keypoints_aug[1], y2=keypoints_aug[3])], shape=img.shape)
    new_coords = []
    for kp_idx, keypoint in enumerate(keypoints_aug.keypoints):
        keypoint_old = keypoints.keypoints[kp_idx]
        x_old, y_old = keypoint_old.x, keypoint_old.y
        x_new, y_new = keypoint.x, keypoint.y
        #print("[Keypoints for image #%s] before aug: x=%s y=%s | after aug: x=%s y=%s" % (img, x_old, y_old, x_new, y_new))
        new_pair = (int(x_new), int(y_new))
        new_coords.append(new_pair)
    print("new_coords: ", new_coords)
    #bbs_aug = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=new_coords[0], x2=new_coords[2], y1=new_coords[1], y2=new_coords[3])], shape=img.shape)
    return new_coords

def back_forward_convert(coordinate, with_label=False):
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
        #for rect in coordinate:
        box = np.int0(coordinate)
        print("box: ", box)
        box = box.reshape([4, 2])
        rect1 = cv2.minAreaRect(box)

        x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        boxes.append([y, x, h, w, theta])

    return np.array(boxes, dtype=np.float32), rect1

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop

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
    img_aug = augment_img(img, d)
    img_open = np.array(Image.open(img))
    frame_box = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=int(1), y1=int(1), x2=int(img_open.shape[1]-1), y2=int(img_open.shape[0]-1))], shape=img_open.shape)
    frame_box = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=int(1), y1=int(1), x2=int(767), y2=int(767))], shape=img_open.shape)
    frame_box = ia.KeypointsOnImage([
        ia.Keypoint(x=0, y=int(img_open.shape[0])),
        ia.Keypoint(x=int(img_open.shape[1]), y=int(img_open.shape[0])),
        ia.Keypoint(x=int(img_open.shape[1]), y=0),
        ia.Keypoint(x=0, y=0)], shape=img_open.shape)
    print("frame_box: ", frame_box)
    frame_box_aug = augment_keypoints_frame(frame_box, img_open, d)
    print("frame_box_aug: ", frame_box_aug)
    frame_box_aug_flat = [int(item) for sublist in frame_box_aug for item in sublist]
    print("frame_box_aug_flat: ", frame_box_aug_flat)
    frame_box_aug_cv, frame_box_aug_cv_rect = back_forward_convert(frame_box_aug_flat)
    print("frame_box_aug_cv: ", frame_box_aug_cv)
    img_crop = crop_minAreaRect(img_aug, frame_box_aug_cv_rect)
    misc.imsave(args.imPath+basename+'_rotate'+str(d)+'_cropFrame'+'.jpg', img_crop)
    pairs = list(zip(*[iter(flat_points_list)] * 2))
    chunked = list(chunks(pairs, 4))
    print("chunked: ", chunked)
    print("pairs: ", pairs)
    #img_aug, chunked_points_aug_list = augment(f, chunked)
    ordered_points_list = []
    bbs_aug_list = []
    for chunk in chunked:
        flat_points_list_chunk = [item for sublist in chunk for item in sublist]
        #flat_points_aug_list_chunk, bbs_aug = augment_keypoints(flat_points_list_chunk, img_aug, d)
        flat_points_aug_list_chunk = augment_keypoints(flat_points_list_chunk, img_aug, d)
        #bbs_aug_list.append(bbs_aug)
        print("flat_points_aug_list_chunk: ", flat_points_aug_list_chunk)
        ordered_points_list_chunk = order_points(flat_points_aug_list_chunk)
        ordered_points_list.append(ordered_points_list_chunk)
    #bbs_aug = bbs_aug_list.on(img_aug)
    #img_bbs_aug = bbs_aug.draw_on_image(img_aug, thickness=2)
    #misc.imsave(args.imPath+basename+'_rotate'+str(d)+'_crop75_bbsRescaled'+'.jpg', img_bbs_aug)
    img_shape = img_aug.shape
    print("image shape: ", img_shape)
    filename = basename
    print("original points list: ", chunked)
    print("ordered_points_list: ", ordered_points_list)
    filename_aug = filename+'_rotate'+str(d)

    #filename_aug = filename    

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

    #print("xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)
    #print("img_xmin, img_xmax, img_ymin, img_ymax: ", img_xmin, img_xmax, img_ymin, img_ymax)

    if xmin < img_xmin:
        out_of_bounds.append(filename_aug)        print("box off frame for file: ", filename_aug)
    elif xmax > img_xmax:
        out_of_bounds.append(filename_aug)
        print("box off frame for file: ", filename_aug)
    elif ymin < img_ymin:
        out_of_bounds.append(filename_aug)
        print("box off frame for file: ", filename_aug)
    elif ymax > img_ymax:
        out_of_bounds.append(filename_aug)
        print("box off frame for file: ", filename_aug)

    return filename_aug, ordered_points_list, img_shape

def run(f):
    for d in [x for x in range(370) if x % 10 == 0][1:]:
        imagename, boxCoords, imageShape = getInfo(f, d)
        XML(imagename, boxCoords, imageShape, d)
    return

out_of_bounds = []

for f in glob.glob(args.annPath+'*.xml'):
    run(f)

#print("out_of_bounds: ", out_of_bounds)
out_of_bounds_outfile = open('out_of_bounds.txt', 'w')
for item in out_of_bounds:
    out_of_bounds_outfile.write("%s\n" % item)


                                                                                                                                                   350,0-1       Bot

