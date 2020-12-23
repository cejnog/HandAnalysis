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
from sensor_utils import init_device, stop_device, read_frame_from_device
from graphic_utils import segmentImage, show_results

def main():
    # intrinsic paramters of Intel Realsense SR300
    fx, fy, ux, uy = 463.889, 463.889, 320, 240
    # paramters
    dataset = 'icvl'
    if len(sys.argv) == 4:
        bagfile = sys.argv[1]
        dataset = sys.argv[2]
        hand = sys.argv[3]
    lower_ = 1
    upper_ = 650

    # init realsense
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bagfile)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # init hand pose estimation model
    hand_model = ModelPoseREN(dataset,
        lambda img: get_center(img, lower=lower_, upper=upper_),
        param=(fx, fy, ux, uy), use_gpu=True)
    # for msra dataset, use the weights for first split
    if dataset == 'msra':
        hand_model.reset_model(dataset, test_id = 0)
    # realtime hand pose estimation loop
    while True:
        depth = read_frame_from_device(pipeline, depth_scale)
        # preprocessing depth
        #depth[0:100, :] = 0
        #depth[540:640, :] = 0
        depth[depth == 0] = depth.max()
        #depth = segmentImage(depth, upper_)
        # training samples are left hands in icvl dataset,
        # right hands in nyu dataset and msra dataset,
        # for this demo you should use your right hand
        #depth = np.minimum(depth, upper_)

        if hand == 'r':
            depth = depth[:, ::-1]  # flip
        
        # get hand pose
        results, _ = hand_model.detect_image(depth)
        img_show = show_results(depth, results, dataset)
        if hand == 'r':
            img_show = img_show[:, ::-1]
        cv2.imshow('result', img_show)
        k = cv2.waitKey(1) 
 
        if k == ord('+'):        
            upper_ += 20
            print(upper_) 
        elif k == ord('-'):        
            upper_ -= 20
            print(upper_)
        elif k == ord('f'):        
            if hand == 'l':
                hand = 'r'
            else:
                hand = 'l'

        if k == ord('q'):
            break
    stop_device(pipeline)

if __name__ == '__main__':
    main()
