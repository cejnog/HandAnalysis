import argparse
from skeleton_utils import read_skeleton_from_file, get_angles, write_angles

njoints = 21
parser = argparse.ArgumentParser(description="Compute measurements from hand tracking results over a camera stream.")
    
parser.add_argument("-s", "--skeleton", type=str, help="Path to skeleton file")
parser.add_argument("-a", "--angle", type=str, help="Path to angle output file")

args = parser.parse_args()

skeleton = args.skeleton

wp = read_skeleton_from_file(skeleton)
angles = get_angles(wp, 0)

write_angles(angles, args.angle)