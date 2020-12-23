import argparse
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import cv2
import pyrealsense2 as rs
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print(BASE_DIR, ROOT_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
from model_pose_ren import ModelPoseREN
import util
from util import get_center_fast as get_center
from sensor_utils import init_device, stop_device, read_frame_from_device, intrinsics
from graphic_utils import drawAngles, nameangles, segmentImage, show_results

# import init_device

#def calibration(upper_, lower_):
    

def main():
    # intrinsic paramters of Intel Realsense SR300
    fx, fy, ux, uy = 463.889, 463.889, 320, 240
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-d", "--dataset", type=str, help="Dataset used")
    parser.add_argument("-hand", "--hand", type=str, help="Whether the hand is (l)eft or (r)ight")
    parser.add_argument("-o", "--outputbag", type=str, help="Path to the resulting bag file")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    
    
    # paramters
    dataset = 'icvl'
    if args.dataset:
        dataset = args.dataset
    if args.hand:
        hand = args.hand
    if args.outputbag:
        output = args.outputbag
    lower_ = 1
    upper_ = 450
    segment = False
    # init realsense
    pipeline = rs.pipeline()
    config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    print 'config'
    # Start streaming
    #intr = intrinsics(pipeline, config)
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()    # init hand pose estimation model
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    hand_model = ModelPoseREN(dataset,
        lambda img: get_center(img, lower=lower_, upper=upper_),
        param=(fx, fy, ux, uy), use_gpu=True)
    # for msra dataset, use the weights f   or first split
    if dataset == 'msra':
        hand_model.reset_model(dataset, test_id = 0)
    angle = 0
    showAngles = False
    # realtime hand pose estimation loop
    while True:
        depth = read_frame_from_device(pipeline, depth_scale)
        # preprocessing depth
        
        
        depth[depth == 0] = depth.max()
        # training samples are left hands in icvl dataset,
        # right hands in nyu dataset and msra dataset,
        # for this demo you should use your right hand
        if hand == 'l':
            depth = depth[:, ::-1]  # flip
        # get hand pose
        # if segment:
        #     depth = segmentImage(depth, upper_)
        
        results, _ = hand_model.detect_image(depth)
        

        if showAngles:
            if hand == 'l':
                img_show = img_show[:, ::-1].copy()  # flip
            
            img_show = drawAngles(depth, results, dataset, nameangles[angle], depth_intrin)
            if hand == 'l':
                img_show = img_show[:, ::-1]  # flip
        else:
            img_show = show_results(depth, results, dataset)

        if hand == 'l':
            img_show = img_show[:, ::-1, :]  # flip
                    

        # img_show = drawAngles(depth, results, dataset)
        # todo: invert text?
        img_show = cv2.resize(img_show, (1280, 960))
        cv2.imshow('result', img_show)
        k = cv2.waitKey(1) 
 
        if k == ord('='):        
            upper_ += 50
            print(upper_) 
        elif k == ord('-'):        
            upper_ -= 50
            print(upper_)
        elif k == ord('g'):        
            segment = not(segment)
        elif k == ord('s'):        
            # pipeline.stop()
            break
        elif k == ord('f'):        
            if hand == 'l':
                hand = 'r'
            else:
                hand = 'l'

        elif k == ord('r'):
            pipeline.stop()
            config.enable_record_to_file(output)
            profile = pipeline.start(config)
            print("Started recording. " + str(hand) + " hand.")            
        
        elif k == ord('l'):        
            angle = (angle + 2) % 20 - 1
        elif k == ord('k'):        
            showAngles = not(showAngles)
        elif k == ord('q'):
            break 
         
    stop_device(pipeline)

if __name__ == '__main__':
    main()
