import cv2
from enum import Enum
import xarray as xr
from graphic_utils import skeletonMeasures, nameangles, flexionangles, abductionangles
import numpy as np
import os
from scipy import signal
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def read_skeleton_from_file(filename, smooth=0):
    njoints = 21
    f = open(filename, 'r')
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
    if smooth != 0:
        for i in range(njoints):
            for j in range(3):
                wp[:,i,j] = signal.savgol_filter(wp[:,i,j], int(smooth), 3)
        
    return wp


def get_angles(wp, smooth):
    from sensor_utils import intrinsics
    njoints = 21
    if smooth != 0:
        for i in range(njoints):
            for j in range(3):
                wp[:,i,j] = signal.savgol_filter(wp[:,i,j], int(smooth), 3)
        
    angle = np.ndarray(shape=(wp.shape[0], len(nameangles)), dtype=float)
    intr = intrinsics()
    for i in range(wp.shape[0]):
        x, y = skeletonMeasures(wp[i], 'hands17', intr)
        for item in range(len(x)):
            angle[i,item] = x[nameangles[item]]        
    return angle

def write_angles(angles, filename):
    f = open(filename, 'w')
    for angle in range(angles.shape[0]):
        for joint in range(angles.shape[1]):
            f.write('{:.3f} '.format(angles[angle, joint]))    
        f.write("\n")
    f.close()

def write_skeleton(skeleton, filename):
    f = open(filename, 'w')
    s = skeleton.shape
    for timeframe in range(s[0]):
        minx = np.min(skeleton[timeframe, :, 0])
        maxx = np.min(skeleton[timeframe, :, 1])
        f.write('{:.6f} {:.6f} {:.6f} {:.6f}'.format(np.min(skeleton[timeframe, :, 0]), np.max(skeleton[timeframe, :, 0]), 
            np.min(skeleton[timeframe, :, 1]), np.max(skeleton[timeframe, :, 1])))
        f.write("\n")
        for joint in range(s[1]):
            f.write('{:.3f} {:.3f} {:.3f}'.format(skeleton[timeframe, joint, 0], skeleton[timeframe, joint, 1], skeleton[timeframe, joint, 2]))
            f.write("\n")
        f.write("\n")
    f.close()

def read_angles_from_file(filename):
    f = open(filename, 'r')
    angles = list()
    while True:
        line = f.readline().split(' ')        
        listframe = list()
        for x in range(len(nameangles)):
            listframe.append(float(line[x]))
        angles.append(listframe)
        if f.tell() == os.fstat(f.fileno()).st_size:
            break
    f.close()
    return xr.DataArray(angles)
    
def get_flexion_landmarks(angle, smooth=25, peaknr=5, offset=3):
    samplesize = angle.shape[0]
    #smooth
    if samplesize >= 25:
        for i in range(len(flexionangles)):
            angle[:,i] = signal.savgol_filter(angle[:, i], int(smooth), 3)
        
    numpeaks = np.zeros(shape=(len(angle)))
        
    #for each type of measurement
    for i in range(len(flexionangles)):    
        absmin = np.argmin(angle[:,i])
        #get local minima
        minidxs = signal.argrelextrema(angle[:,i].values, np.less)
        #for each local minima
        for j in minidxs[0]:
            # print(j)
            #test whether j is close to the global minima (offset is a parameter)
            if np.abs(angle[j, i] - angle[absmin, i]) < 3*offset:
                numpeaks[j] += 1
            #get points in a small window around j
            for k in range(j - 3, j + 3):
                #if such points are valid and close enough to the global minima they are also considered peaks
                if k >= 0 and k < len(angle):
                    if np.abs(angle[k, i] - angle[j, i]) < offset and np.abs(angle[k, i] - angle[absmin, i]) < 3*offset:
                        numpeaks[k] += 1
    #at the end of this loop, numpeaks contains a peak histogram of the signal

    #parameter peaknr is used as a threshold for this histogram: if points are peaks in peaknr measurements, then most ]
    # likely they represent a 'rest' position
    landmarks = list()
    j = 0
    n1 = 0
    n2 = 0
    while j < samplesize:
        # walks in the samples looking for points that are beginnings of 'unrest' sequences
        if numpeaks[j] < peaknr and numpeaks[j-1] >= peaknr:
            n1 = j
            n2 = -1
            #walk in the sequence looking for the next rest point
            while j < samplesize and n2 == -1:
                # print(j, numpeaks[j])
                if numpeaks[j] < peaknr:
                    j = j+1
                else:
                    n2 = j
            #print(n1, n2)
            #if the sequence is long enough and the peak of the sequence satisfies certain conditions (average eccentricity), the landmark is added
            #average eccentricity is between 50 and 70.
            # (less than 15 frames is considered an outlier)

            if n2 - n1 > 15:
                if j == samplesize:                 
                    landmarks.append([n1, j])    
                else:
                    eccen = 0
                    for i in range(len(flexionangles)):
                        maxv = np.max(angle[n1:n2,i].values)
                        minv = np.min(angle[n1:n2,i].values)
                        eccen += maxv - minv
                    if eccen / len(flexionangles) > 50 and eccen / len(flexionangles) < 70:
                        landmarks.append([n1, n2])
                    print(eccen / len(flexionangles))
        else:
            j = j+1
            # print(j, numpeaks[j])
    print(landmarks)
    return landmarks

def get_flexion_landmarks_avg(angle, smooth=75, threshold=0.8):
    avg = np.average(angle.values, axis=1)    
    avg = signal.savgol_filter(avg, int(smooth), 3)
    difference = np.diff(avg)
    minidxs = signal.argrelextrema(avg, np.less)
    maxidxs = signal.argrelextrema(avg, np.greater)
    #plt.plot(avg)
    #plt.plot(difference)
    landmarks = list()
    n1 = 1
    n2 = minidxs[0][0]
    if n2 - n1 > 15:
        print(n1, n2)
        while (np.abs(difference[n1]) < threshold) and n1 < len(difference)-1:
            n1 = n1 + 1
        
        while (np.abs(difference[n2]) < threshold) and n2 > n1+15:
            n2 = n2 - 1
        print(n1, n2)
        landmarks.append([n1, n2])
    for val in range(len(minidxs[0])-1):
        n1 = minidxs[0][val]
        n2 = minidxs[0][val+1]
        if n2 - n1 > 15:
            print(n1, n2)
            while (np.abs(difference[n1]) < threshold) and n1 < len(difference)-1:
                n1 = n1 + 1
            
            while (np.abs(difference[n2]) < threshold) and n2 > n1+15:
                n2 = n2 - 1
            #plt.scatter(n1, avg[n1])
            #plt.scatter(n2, avg[n2])
            print(n1, n2)
            if n1 < n2:
                landmarks.append([n1, n2])
    #plt.show()
    return landmarks


def get_flexion_landmarks_derivs(angle, smooth=5, peaknr=5, offset=3):
    samplesize = angle.shape[0]
    for i in range(len(flexionangles)):
        angle[:,i] = signal.savgol_filter(angle[:, i], int(smooth), 3)
    
    numpeaks = np.zeros(shape=(len(angle)))
        
    for i in range(len(flexionangles)):    
        absmin = np.argmax(angle[:,i])
        maxidxs = signal.argrelextrema(angle[:,i].values, np.greater)
        for j in maxidxs[0]:
            # print(j)
            if np.abs(angle[j, i] - angle[absmin, i]) < 3*offset:
                numpeaks[j] += 1
            for k in range(j - 3, j + 3):
                if k >= 0 and k < len(angle):
                    if np.abs(angle[k, i] - angle[j, i]) < offset and np.abs(angle[k, i] - angle[absmin, i]) < 3*offset:
                        numpeaks[k] += 1
            

    landmarks = list()
    j = 0
    n1 = 0
    n2 = 0
    while j < samplesize:
        # print(j, numpeaks[j], n1, n2)
        if numpeaks[j] < peaknr and numpeaks[j-1] >= peaknr:
            n1 = j
            n2 = -1
            while j < samplesize and n2 == -1:
                # print(j, numpeaks[j])
                if numpeaks[j] < peaknr:
                    j = j+1
                else:
                    n2 = j
            if n2 - n1 > 5:
                if j == samplesize:                 
                    landmarks.append([n1, j])    
                else:
                    eccen = 0
                    for i in range(len(flexionangles)):
                        maxv = np.max(angle[n1:n2,i].values)
                        minv = np.min(angle[n1:n2,i].values)
                        eccen += maxv - minv
                    if eccen / len(flexionangles) > 10:
                        landmarks.append([n1, n2])
        else:
            j = j+1
            # print(j, numpeaks[j])
    print(landmarks)
    return landmarks


class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)

def show_skeleton(pose):
    img = np.ones((480, 640))
    # img = np.minimum(img, 650)
    #img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    colors = [Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED,
              Color.GREEN, Color.GREEN, Color.GREEN,
              Color.BLUE, Color.BLUE, Color.BLUE,
              Color.YELLOW, Color.YELLOW, Color.YELLOW,
              Color.PURPLE, Color.PURPLE, Color.PURPLE,
              Color.RED, Color.RED, Color.RED]
    colors_joint = [Color.CYAN, Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.PURPLE, Color.PURPLE, Color.PURPLE,
                Color.RED, Color.RED, Color.RED]
    sketch_setting = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11), (3, 12), (12, 13), (13, 14), (4, 15), (15, 16),
                (16, 17), (5, 18), (18, 19), (19, 20)]
    idx = 0
    #plt.figure()
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, colors_joint[idx].value, -1)
        #plt.scatter(pt[0], pt[1], pt[2])
        idx = idx + 1
    idx = 0
    for x, y in sketch_setting:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), colors[idx].value, 2)
        idx = idx + 1
    return img
