'''
Author: Luciano Cejnog
Institution: IME-USP
Release Date: 2020-12-29
'''

import cv2
import sys, os, shutil
#import logging
#logging.basicConfig(level=logging.INFO)
import numpy as np
import pyrealsense2 as rs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
from model_pose_ren import ModelPoseREN
import util
from util import get_center_fast as get_center
import argparse
import os.path
from tqdm import tqdm
from graphic_utils import segmentImage, show_results


from numpy import array, int32

def pgmread(filename):
    """  This function reads Portable GrayMap (PGM) image files and returns
    a numpy array. Image needs to have P2 or P5 header number.
    Line1 : MagicNum
    Line2 : Width Height
    Line3 : Max Gray level
    Lines starting with # are ignored """
    f = open(filename,'r')
    # Read header information
    count = 0
    while count < 3:
        line = f.readline()
        if line[0] == '#': # Ignore comments
            continue
        count = count + 1
        if count == 1: # Magic num info
            magicNum = line.strip()
            if magicNum != 'P2' and magicNum != 'P5':
                f.close()
                print 'Not a valid PGM file'
                exit()
        elif count == 2: # Width and Height
            [width, height] = (line.strip()).split()
            width = int(width)
            height = int(height)
        elif count == 3: # Max gray level
            maxVal = int(line.strip())
    # Read pixels information
    img = []
    buf = f.read()
    elem = buf.split()
    if len(elem) != width*height:
        print 'Error in number of pixels'
        exit()
    for i in range(height):
        tmpList = []
        for j in range(width):
            tmpList.append(elem[i*width+j])
        img.append(tmpList)
    return (array(img), width, height)

#Show results with pose
def show_results_upper(img, results, dataset, upper_, lower_ = 0):
    img = np.minimum(img, 1500)
    #img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_show = util.draw_pose(dataset, img, results)
    return img

#def processFile(img, hand='l', upper_=400):

#    return ((img_show, results))

upper_ = 400
lower_ = 1
dataset = 'hands17'
fx, fy, ux, uy = 463.889, 463.889, 320, 240
cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
folder = sys.argv[1]
hand = 'l'# if 'l' in folder else 'r'
i = 0
hand_model = ModelPoseREN(dataset,
        lambda img: get_center(img, lower=lower_, upper=upper_),
        param=(fx, fy, ux, uy), use_gpu=True)
        
while True:
    if os.path.exists(folder + '/' + str(i) + '.pgm'):
        img, w, h = pgmread(folder + str(i) + '.pgm')
        img = np.asarray(img, dtype=np.float32)
            
        print(img.min(), img.max())
        if hand == 'r':
            img = img[:, ::-1]  # flip

        #detect joints using Pose-REN
        results, _ = hand_model.detect_image(img) 
        
        #Show skeleton over image
        img = np.uint8(img)
        img[img == 0] = img.max()
        img_show = show_results_upper(img, results, dataset, upper_)
        cv2.imshow("Depth Stream", img_show)
        k = cv2.waitKey(1) 
        
        #Control limit max depth using + and - keys
        if k == ord('+'):        
            upper_ += 20
            print(upper_) 
        elif k == ord('-'):        
            upper_ -= 20
            print(upper_)

        cv2.waitKey(30)

    else:
        exit()
    i += 1

#print(sorted(os.listdir(folder)))/
