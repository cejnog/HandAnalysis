import os, sys
import numpy as np
import cv2
import pyrealsense2 as rs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print(BASE_DIR, ROOT_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
import util

def init_device():
    # Configure depth streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    print('config')
    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " + depth_scale)
    return pipeline, depth_scale

def stop_device(pipeline):
    pipeline.stop()
    
def read_frame_from_device(pipeline, depth_scale, color=False):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if color:
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
    #if not depth_frame:
    #    return None
    # Convert images to numpy arrays
    depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)    
    depth = depth_image * depth_scale * 1000
    if color:
        return depth, color_image
    else:
        return depth


def read_color_frame_from_device(pipeline, depth_scale):
    frames = pipeline.wait_for_frames()
    return frames.get_color_frame()


def intrinsics(pipeline=None):
    if pipeline == None:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, 'example.bag')
        cfg = pipeline.start(config) # Start pipeline and get the configuration it found    
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    
        pipeline.stop()
    else:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    
    return depth_intrin
    

def readbag(bagfile, hand='l', dataset='icvl'):
    pipeline = rs.pipeline()      
    # Create a config object
    config = rs.config()
    
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, bagfile, repeat_playback = False)
    depths = list()
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

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
                
        #depth[depth == 0] = depth.max()
        if hand == 'l':
            depth = depth[:, ::-1]  # flip

        if dataset == 'icvl':
            depth = depth[:, ::-1]  # flip

        depths.append(depth)
        
    return depths
