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

import cv2
import sys, os, shutil

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

def show_results(img):
    img = np.minimum(img, 450)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    return img
        

def main():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    parser.add_argument("-o", "--output", type=str, help="Path to the video file (avi)")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()
    if not args.output:
        output = os.path.splitext(args.input)[0] + ".avi"
    else:
        if os.path.splitext(args.output) != ".avi":
            print("The given file is not of correct file format.")
            print("Only .avi files are accepted")
            exit()
        output = args.output

    colorFrame0 = None
    
    # intrinsic paramters of Intel Realsense SR300
    fx, fy, ux, uy = 463.889, 463.889, 320, 240
    # paramters
    # dataset = 'icvl'
    # if len(sys.argv) == 3:
    #     dataset = sys.argv[2]
    colorizer = rs.colorizer()
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    lower_ = 1
    upper = 650    
    videoFile = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,480))
    i=0
    
    pipeline = rs.pipeline()
    
    
    # Create a config object
    config = rs.config()
    
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    
    # Configure the pipeline to stream the depth stream
    
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    print("Q")
    # Start streaming from file
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    
    
    
    while True:
        
        # depth = read_frame_from_device(pipeline, depth_scale)
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        
        color = frames.get_color_frame()
        frame = frames.get_depth_frame()
        # frame = decimation.process(frame)
        frame = depth_to_disparity.process(frame)
        # frame = spatial.process(frame)
        # frame = temporal.process(frame)
        frame = disparity_to_depth.process(frame)

        # frame = hole_filling.process(frame)    
        # if not depth_frame:
        #    return None
        # Convert images to numpy arrays
        
        depth_image = np.asarray(frame.get_data(), dtype=np.float32)
        color_image = np.asanyarray(color.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (640,480))

        if np.array_equal(colorFrame0, None):
            colorFrame0 = color_image
        else:
            if (np.array_equal(color_image, colorFrame0)):
                pipeline.stop()
                break
        
        depth = depth_image * depth_scale * 1000        
        
        
        # preprocessing depth
        depth[depth == 0] = depth.max()
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = show_results(depth)
        # cv2.cvtColor(depth_colormap, cv2.COLOR_GRAY2RGB, depth_colormap)
        images = np.hstack((color_image, depth_colormap))
        # images = show_results(depth)  #depth_colormap
        # training samples are left hands in icvl dataset,
        # right hands in nyu dataset and msra dataset,
        # for this demo you should use your right hand        
        # cv2.imshow('result', images)
        # cv2.waitKey(30)
        videoFile.write(images)        
        # videoFile.write(img_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i=i+1
    # stop_device(pipeline)
    videoFile.release()
    


if __name__ == '__main__':
    main()