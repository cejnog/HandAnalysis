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
        

def show_results(img, results, dataset, upper_, lower_ = 0):
    img = np.minimum(img, upper_)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_show = util.draw_pose(dataset, img, results)
    return img_show

def processBag(hand_model, upper_, input, output, dataset, hand, jointsFile): 
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
        if hand == 'l':
            depth = depth[:, ::-1]  # flip

        if dataset == 'icvl':
            depth = depth[:, ::-1]  # flip

        results, _ = hand_model.detect_image(depth)  
        resultlist.append((depth, results))
        img_show = show_results(depth, results, dataset, upper_)
        
        
    print("Terminou de processar")
    # #step 2 (critical): process each depth image, getting results
    # # i = 0
    # for i in tqdm(range(len(depths))):    
    #     results, _ = hand_model.detect_image(depths[i])    
    #     resultlist.append((depths[i], results))
    
    #step 3: show results (and save in video)
    frame_width=640
    frame_height=480
    movement = list()
    videoFile = cv2.VideoWriter(output,cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
    for r in resultlist:
        img_show = show_results(r[0], r[1], dataset, upper_)
        if hand == 'l':
            img_show = img_show[:, ::-1]  # flip
        videoFile.write(img_show)        
        #dlist.append(img_show)
    videoFile.release()
    
    print("Terminou de gravar o video")
    #step 4: save joints file
    f = open(jointsFile, 'w')
    for r in resultlist:
        results = r[1]
        f.write('{:3f} {:.3f} {:.3f} {:.3f}\n'.format(np.min(results[:,0]), np.max(results[:,0]), np.min(results[:,1]), np.max(results[:,1])))
        for joint in results:
            f.write('{:.3f} {:.3f} {:.3f}\n'.format(joint[0], joint[1], joint[2]))    
        f.write("\n")
        movement.append(results)        
    f.close()          
    print("Atualizou movimentos")

def main():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset used")
    parser.add_argument("-hand", "--hand", type=str, help="Whether the hand is (l)eft or (r)ight")
    parser.add_argument("-o", "--output", type=str, help="Path to the video file (avi)")
    parser.add_argument("-j", "--jointsFile", type=str, help="Path to the output joints file (txt)")
    parser.add_argument("-s", "--segment", type=str, help="If segmentation is applied")
    parser.add_argument("-u", "--upper", type=int, help="Segmentation threshold")
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
        output = args.output
        if os.path.splitext(args.output)[1] != ".avi":
            print("The given file is not of correct file format.")
            output = os.path.splitext(args.input)[0] + ".avi"
            print("Using " + output + " instead")
            print("Only .avi files are accepted")            
        
    if not args.jointsFile:
        jointsFile = os.path.splitext(args.input)[0] + ".txt"
    else:
        jointsFile = args.jointsFile
        if os.path.splitext(args.jointsFile)[1] != ".txt":
            print("The given file is not of correct file format.")
            jointsFile = os.path.splitext(args.input)[0] + ".txt"
            print("Using " + jointsFile + " instead")        

    if not args.hand:
        print("No hand parameter given. Using left hand instead.")
        hand = 'l'
    else:
        if args.hand == 'l' or args.hand == 'r':
            hand = args.hand
        else:
            print("Unrecognized option for parameter --hand. Using left hand instead.")
            hand = 'l'

    if not args.dataset:
        print("Unrecognized option for parameter --dataset. Using ICVL dataset.")
        dataset = 'icvl'
    else:
        if args.dataset == 'icvl' or args.dataset == 'nyu' or args.dataset == 'msra' or args.dataset == 'hands17':
            dataset = args.dataset
        else:
            print("Unrecognized option for parameter --dataset. Using ICVL dataset.")
            dataset = 'icvl'
    segment = True
    if not args.segment:
        segment = False
    if not args.upper:
        upper_ = 350
    else:
        upper_ = args.upper
        
    fx, fy, ux, uy = 463.889, 463.889, 320, 240
    temporal = rs.temporal_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    lower_ = 0
    hand_model = ModelPoseREN(dataset,
        lambda img: get_center(img, lower=lower_, upper=upper_),
        param=(fx, fy, ux, uy), use_gpu=True)
    # for msra dataset, use the weights for first split
    if dataset == 'msra':
        hand_model.reset_model(dataset, test_id = 0)

    #print(args.input, output, dataset, hand, jointsFile)
    processBag(hand_model, upper_, args.input, output, dataset, hand, jointsFile)
    
if __name__ == '__main__':
    main()
