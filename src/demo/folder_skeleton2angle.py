import sys, os
from skeleton_utils import read_skeleton_from_file, get_angles, write_angles

s_dir = "/home/cejnog/Documents/Pose-REN/data/07-2019/results/original/hands17/skeletons"
a_dir = "/home/cejnog/Documents/Pose-REN/data/07-2019/results/original/hands17/angles"

for in_dir in sorted(os.listdir(s_dir)):
    for file in sorted(os.listdir(s_dir + "/" + in_dir)):
        sf = s_dir + "/" + in_dir + "/" + file
        af = a_dir + "/" + in_dir + "/" + file[:9] + ".txt"
        if os.path.isfile(sf):
            skeleton = read_skeleton_from_file(sf)
            angles = get_angles(skeleton, 0)
            write_angles(angles, af)
            #print(af)
