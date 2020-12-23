import xarray as xr
import sys, os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sensor_utils import readbag
import argparse
from scipy import signal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
from model_pose_ren import ModelPoseREN
import util
from util import get_center_fast as get_center
from util import get_sketch_setting
import matplotlib.gridspec as gridspec
from graphic_utils import drawAngles, nameangles, namejoints, show_results

def nothing(x):
    pass

def unit_vector(vector):
    """ qReturns the unit vector of the vector.  """
    return np.array(vector / np.linalg.norm(vector))

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def skeletonMeasures(skeleton, dataset):
    if dataset == 'icvl':
        njoints = 16
    elif dataset == 'nyu':
        njoints = 14
    elif dataset == 'msra':
        njoints = 21        
    elif dataset == 'hands17':
        njoints = 21        
    sketch = get_sketch_setting(dataset)
    skel = list()
    for i in range(njoints):
        skel.append(list())
    for joint in sketch:
        skel[joint[0]].append(joint[1])
    
    measures = [[0,[1],6],[0,[2],8],[0,[3],11],[0,[4],14],[0,[5],17],
                [1,[6],7],[2,[9],10],[3,[12],13],[4,[15],16],[5,[18],19],
                [6,[7],8],[9,[10],11],[12,[13],14],[15,[16],17],[18,[19],20],
                [7,[6,2],9],[9,[2,3],12],[12,[3,4],15],[15,[4,5],18]]
    
    j = 0
    angle = np.zeros((len(measures)))
    positions = np.ndarray(shape=(len(measures),3))
    for measure in measures:
        if len(measure[1]) == 2:
            midpoint = 0.5 * (skeleton[measure[1][0]] + skeleton[measure[1][1]])
        else:
            midpoint = skeleton[measure[1][0]]
        v1 = midpoint - skeleton[measure[0]]
        v2 = midpoint - skeleton[measure[2]]
        # print(v1, v2, measure[0], measure[1], measure[2])
        angle[j] = angle_between(v1, v2)
        positions[j] = midpoint
        j = j+1
    
    
    # mid = [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 10, 13, 16, 19]
    # angle = np.zeros((njoints))
    # positions = np.ndarray(shape=(njoints,3))
    
    # roots = [0,1,2,3,4,5,6,9,12,15,18]
    # for root in roots:
    #     for edge in skel[root]:
    #         v1 = skeleton[edge] - skeleton[root]    
    #         for edge2 in skel[edge]:
    #             v2 = skeleton[edge] - skeleton[edge2]    
    #             # print(v1, v2)
    #             angle[edge] = angle_between(v1, v2)
    #             positions[edge] = skeleton[edge]

    # angle[19]
    #angles = 'abduction'
    #measures = [[0,0,0], [0,0,0], [0,2,3], [0,3,4], [0,4,5], [0,5,6], [0,0,0], [0,7,9], [0,9,12], [0,12,15], [0,15,18],
    #[0,0,0], [0,8,10], [0,10,13], [0,13,16], [0,16,19], [0,0,0], [0,8,11], [0,11,14], [0,14,17], [0,17,20]]
    #j = 0
    #for measure in measures:
    #    if measure[0] == measure[1]:
    #        angle[j] = 0
    #    else:
    #        v1 = skeleton[measure[0]] - skeleton[measure[1]]
    #        v2 = skeleton[measure[0]] - skeleton[measure[2]]            
    #        angle[j] = angle_between(v1, v2)
    #    j = j+1
    return angle, positions

def main():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    

    parser.add_argument("-f", "--inputfile", type=str, help="Path to the joint file")
    parser.add_argument("-b", "--bagfile", type=str, help="Path to the bag file")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset used to train model (nyu/msra/icvl)")
    parser.add_argument("-s", "--smooth", type=str, help="Filter?")
    parser.add_argument("-hand", "--hand", type=str, help="Whether the hand is (l)eft or (r)ight")
    # parser.add_argument("-o", "--outputfile", type=str, help="Output joint file")
    args = parser.parse_args()

    # outputfile = args.outputfile
    dataset = args.dataset
    depths = readbag(args.bagfile, args.hand, args.dataset)
    if dataset == 'icvl':
        njoints = 16
    elif dataset == 'nyu':
        njoints = 14
    elif dataset == 'msra':
        njoints = 21        
    elif dataset == 'hands17':
        njoints = 21        
    
    
    
    
    path = args.inputfile
    print(path)
    
    f = open(path, 'r')
    i = 0    
    movement = list()
    while True:        
        bbox = f.readline().split(" ")
        bbox_center_x = 0.5*(float(bbox[0]) + float(bbox[1]))
        bbox_center_y = 0.5*(float(bbox[2]) + float(bbox[3]))
        joints = np.ndarray(shape=(njoints, 3), dtype=float)
        for i in range(njoints):
            line = f.readline()
            # joints[i] = ([float(line.split(' ')[0])-bbox_center_x, float(line.split(' ')[1])-bbox_center_y, float(line.split(' ')[2])])
            joints[i] = ([float(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[2])])
        next = f.readline()
        movement.append(joints)
        if f.tell() == os.fstat(f.fileno()).st_size:
            break
        
    wp = xr.DataArray(movement)
    y = np.ndarray(shape=(njoints,3))
    if args.smooth:
        for i in range(njoints):
            for j in range(3):
                wp[:,i,j] = signal.savgol_filter(wp[:,i,j], int(args.smooth), 3)
    
    
    # points = list()
    # for i in range(njoints):
    #     points.append(list())
    #     for j in range(3):
    #         points[i].append(list())
    #         k=0
    #         peaks = signal.find_peaks_cwt(wp[:,i,j], np.arange(1, 20))
    #         for peak in peaks:
    #             points[i][j].append(peak)
    #         peaks2 = signal.find_peaks_cwt(-wp[:,i,j], np.arange(1, 20))
    #         for peak in peaks2:
    #             points[i][j].append(peak)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # #print(points)
    # saliency = np.ndarray((len(depths)))
    # for i in range(len(depths)):
    #     saliency[i] = 0
    #     for k in range(njoints):
    #         for j in range(3):
    #             if i in points[k][j]:
    #                 saliency[i] += 1
    #plt.plot(saliency)
    #plt.show()
    angle = np.ndarray(shape=(len(depths), njoints), dtype=float)
    positions = np.ndarray(shape=(len(depths), njoints, 3), dtype=float)
    sum_angles = list()
    for i in range(len(depths)):
        x, y = skeletonMeasures(wp[i], args.dataset)
        for item in range(len(x)):
            angle[i, item] = x[item]
            positions[i, item, 0] = y[item][0]
            positions[i, item, 1] = y[item][1]
            positions[i, item, 2] = y[item][2]
        #while True:
        #i = cv2.getTrackbarPos("Frame", "Colorbars")

    fig2 = plt.figure(figsize=(25, 20))
    fig2.tight_layout()
    spec2 = gridspec.GridSpec(nrows=3, ncols=njoints/3)
    points = list()
    print(angle)
    for i in range(len(nameangles)):
        # print(len(angle[:][i]))
        f = fig2.add_subplot(spec2[i%3, i/3])
        f.set_title(nameangles[i])
        f.plot(angle[:,i], label=str(i))
        
    # plt.legend()
    # fig, axs = plt.subplots(2,1)
    # axs[0].plot(saliency)
    
    # axs[1].plot(sum_angles)
    plt.show()

    cv2.namedWindow('Colorbars')
    cv2.createTrackbar("Frame", "Colorbars",0,len(depths),nothing)  
    printtext = ['names', 'angles', 'jointids']
    angletypes = ['A', 'F', '']
    selectedindex = 0
    selectedtype = 0
    intr = intrinsics()
    while True:
        i = cv2.getTrackbarPos("Frame", "Colorbars")
        # for i in range(len(depths)):
        img = drawAngles(depths[i], wp[i], args.dataset, nameangles[selectedindex], intr)
        # for point in range(len(nameangles)):            
        #     if printtext[selectedindex] == 'names':
        #         text = nameangles[point]
        #     if printtext[selectedindex] == 'angles':
        #         if nameangles[point].startswith(angletypes[selectedtype]):
        #             text = str("%.1f" % angle[i][point])
        #         else:
        #             text = ''
        #     if printtext[selectedindex] == 'jointids':
        #         text = str(point)
        #     print(text, (int(positions[i,point,0]),int(positions[i,point,1])))
        #     cv2.putText(img, text, (int(positions[i,point,0]),int(positions[i,point,1])), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200),2,cv2.LINE_AA)
        # print()
        cv2.imshow('Colorbars', img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        elif key == ord('f'):
            selectedindex = np.remainder((selectedindex + 1), len(nameangles))
        # elif key == ord('g'):
        #     selectedtype = np.remainder((selectedtype + 1), 3)
    

main()
plt.show()
