from dataset3_utils import *
import random
from skeleton_utils import read_skeleton_from_file, read_angles_from_file, write_skeleton, get_angles, write_angles
import xarray as xr
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import cv2
from data_augmentation import genRandomSkeleton
import shutil
import sys

sigmas = [1, 2, 4]
path = "data/11-2019/loo-clean/"

if len(sys.argv) == 1:
    l1 = patients
    l2 = patient_skeletons
else:
    if sys.argv[1] != 'p':
        l1 = control
        l2 = control_skeletons
for sigma in sigmas:
    for hand in ['l', 'r']:
        for patient in sorted(l1):
            for i in range(100):
                #print(patient)
                filelist = sorted([x for x in l2 if x.startswith(patient) and x[5] == hand and x[4] == 'f'])
                print(filelist)
                if len(filelist) != 0:
                    nr  = random.randrange(0, len(filelist))
                    key = filelist[nr]
                    print(key)
                    newkey = patient + "_f" + hand + str(i) + "_" + filelist[nr].split("_", 1)[1]
                    filepath = path + "skeletons/s" + str(sigma) + "/" + newkey + ".txt"
                    angpath = path + "angles/s" + str(sigma) + "/" + newkey + ".txt"
                    print(key, newkey, filepath)
                    s_dir = path + "skeletons/s" + str(sigma) + "/"
                    a_dir = path + "angles/s" + str(sigma) + "/"
                    land = random.choice(landmarks[key])
                    skeleton = genRandomSkeleton(l2, key, filepath, sigma)
                    angles = get_angles(skeleton, 0)
                    write_angles(angles[land[0]:land[1], :], angpath)