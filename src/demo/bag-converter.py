import xarray as xr
import cv2
import sys, os, shutil
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import pyrealsense2 as rs
import os
import sys
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
#from sensor_utils import read_bag
import cv2
import sys, os, shutil
from tqdm import tqdm
from graphic_utils import drawAngles, graphicAngles, namejoints, nameangles, segmentImage
import dataset3_utils
from PIL import Image

from numpy import array, int32
import numpy as np
import struct

def pgm_binary_write ( img, filename, maxval ):

    img = int32(img).tolist()
    f = open(filename,'w')
    width = 0
    height = 0
    for row in img:
        height = height + 1
        width = len(row)
    file_handle = open ( filename, 'wb' )
    #
    #  Set up the header.
    #
    pgm_header = "P5 " + str(width) + " " + str(height) + " " + str(maxval) + "\n"
    file_handle.write ( bytearray ( pgm_header, 'ascii' ) ) 
    #
    #  Convert 2D array to 1D vector.
    #
    grayV = np.reshape ( img, width * height )
    #
    #  Pack entries of vector into a string of bytes, replacing each integer
    #  as an unsigned 1 byte character.
    #
    grayB = struct.pack ( '%sB' % len(grayV), *grayV )
    file_handle.write ( grayB )

    file_handle.close ( )

    return

def pgmwrite(img, filename, maxVal=255, magicNum='P2'):
    """  This function writes a numpy array to a Portable GrayMap (PGM) 
    image file. By default, header number P2 and max gray level 255 are 
    written. Width and height are same as the size of the given list.
    Line1 : MagicNum
    Line2 : Width Height
    Line3 : Max Gray level
    Image Row 1
    Image Row 2 etc. """
    img = int32(img).tolist()
    f = open(filename,'w')
    width = 0
    height = 0
    for row in img:
        height = height + 1
        width = len(row)
    f.write(magicNum + '\n')
    f.write(str(width) + ' ' + str(height) + '\n')
    f.write(str(maxVal) + '\n')
    for i in range(height):
        count = 1
        for j in range(width):
            f.write(str(img[i][j]) + ' ')
            if count >= 17:
            # No line should contain gt 70 chars (17*4=68)
            # Max three chars for pixel plus one space
                count = 1
                f.write('\n')
            else:
                count = count + 1
    f.write('\n')
    f.close()

def init_device(inputfile):
    try:
        # Create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()
        # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, inputfile, repeat_playback = False)
        
        # Configure the pipeline to stream the depth stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Start streaming from file
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        return pipeline, depth_scale
    except:
        print("can't init device")

# def show_filtered_depth(img):
#     img = np.minimum(img, 450)
#     img = (img - img.min()) / (img.max() - img.min())
#     img = np.uint8(img*255)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     return img
        

def processBag(input, output, hand='l'): 
    colorFrame0 = None
    i=0
    pipeline = rs.pipeline()
    # Create a config object 
    config = rs.config()
    
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, input, repeat_playback = False)
    
    # Start streaming from file
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    #step 1: depth images are preprocessed and stored into the list
    depths = list()
    resultlist = list()
    hand_model = ModelPoseREN(dataset,
        lambda img: get_center(img, lower=lower_, upper=upper_),
        param=(fx, fy, ux, uy), use_gpu=True)
        
    while True:
        try:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()
        except:
            pipeline.stop()
            break
        frame = frames.get_depth_frame()
        depth_image = np.asarray(frame.get_data(), dtype=np.float32)

        depth = depth_image * depth_scale * 1000        
                
        depth[depth == 0] = depth.max()
        if hand == 'r':
            depth = depth[:, ::-1]  # flip

        depths.append(depth)
    i = 0

    for d in depths:
        result = Image.fromarray(d)
        #bw = result.convert("L")
        pgmwrite(result, output + '/' + str(i) + '.pgm', 1024)
        #result.save(output + '/' + str(i) + '.pgm')
        i += 1

def main(input, hand):
    output ='/media/cejnog/DATA/pgm/' + os.path.splitext(input)[0].split('/')[-1] 
    if not os.path.isdir(output):
        os.mkdir(output)
    processBag(input, output, hand)
    
if __name__ == '__main__':
    for file in sorted(dataset3_utils.bag_files):
        print('/media/cejnog/DATA/pgm/' +  os.path.splitext(dataset3_utils.bag_files[file])[0].split('/')[-1] + '/')
        if os.path.isdir('/media/cejnog/DATA/pgm/' + os.path.splitext(dataset3_utils.bag_files[file])[0].split('/')[-1] + '/'):
            continue

        if os.path.isdir('../../pgm/' + os.path.splitext(dataset3_utils.bag_files[file])[0].split('/')[-1] + '/'):
            continue
        print(file)
        hand = 'l' if 'l' in dataset3_utils.bag_files[file] else 'r'
        main(dataset3_utils.bag_files[file], hand)