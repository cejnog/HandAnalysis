import os, sys
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
import util
from util import get_center_fast as get_center
from util import get_sketch_setting

from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import cv2
import numpy as np
import numpy.linalg

namejoints = ['W', 'I-CMC', 'II-MCP', 'III-MCP', 'IV-MCP', 'V-MCP', 'I-MCP', 
    'I-IP', 'I-TIP', 'II-PIP', 'II-DIP', 'II-TIP', 'III-PIP', 'III-DIP', 'III-TIP', 
    'IV-PIP', 'IV-DIP', 'IV-TIP', 'V-PIP', 'V-DIP', 'V-TIP']
    
nameangles = ['F-I-CMC', 'F-II-MCP', 'F-III-MCP', 'F-IV-MCP', 'F-V-MCP', 'F-I-MCP', 
'F-II-PIP', 'F-III-PIP', 'F-IV-PIP', 'F-V-PIP', 'F-I-IP', 'F-II-DIP', 'F-III-DIP', 'F-IV-DIP', 'F-V-DIP', 
'A-II-TIP', 'A-III-TIP', 'A-IV-TIP', 'A-V-TIP', 'OP-I-MCP', 'OP-II-MCP', 'OP-III-MCP', 'OP-IV-MCP']
    
flexionangles = [x for x in nameangles if x.startswith('F') and not('-I-' in x)]
abductionangles = [x for x in nameangles if x.startswith('A') and not('-I-' in x)]

def unit_vector(vector):
    """ qReturns the unit vector of the vector.  """
    return np.array(vector / np.linalg.norm(vector))

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    # return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1_u, v2_u)), np.dot(v1_u, v2_u)))


def signed_angle_between(v1, v2, normal):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    cross = unit_vector(np.cross(v1_u, v2_u))
    if (np.dot(normal, cross) < 0):
        angle = -angle
    return angle

def angle(p0, midpoint, p1, mult=1):
    v1 = midpoint - p0
    v2 = mult*(p1 - midpoint)
    return angle_between(v1, v2)

def midpoint(p0, p1):
    return 0.5 * (p0 + p1)

def proj(v, n):
    return v - (np.dot(v, n)/np.linalg.norm(n)) * n

def skeletonMeasures(skeleton, dataset, intr):
    if dataset != 'hands17':
        print('not implemented')
        return
    import pyrealsense2 as rs
    njoints = 21
    pskeleton = dict(zip(namejoints, skeleton))
    dskeleton = dict(zip(namejoints, skeleton.copy()))
    for pt in dskeleton:
        l = list()
        l.append(dskeleton[pt][0])
        l.append(dskeleton[pt][1])
        x = rs.rs2_deproject_pixel_to_point(intr, l, dskeleton[pt][2])
        dskeleton[pt][0] = x[0]
        dskeleton[pt][1] = x[1]
        dskeleton[pt][2] = x[2]
    
        
    measures = dict()
    positions = dict()
    
    measures['F-I-CMC'] = angle(dskeleton['W'], dskeleton['I-CMC'], dskeleton['I-MCP'])
    measures['F-II-MCP'] = angle(dskeleton['W'], dskeleton['II-MCP'], dskeleton['II-PIP'])
    measures['F-III-MCP'] = angle(dskeleton['W'], dskeleton['III-MCP'], dskeleton['III-PIP'])
    measures['F-IV-MCP'] = angle(dskeleton['W'], dskeleton['IV-MCP'], dskeleton['IV-PIP'])
    measures['F-V-MCP'] = angle(dskeleton['W'], dskeleton['V-MCP'], dskeleton['V-PIP'])
    measures['F-I-MCP'] = angle(dskeleton['I-CMC'], dskeleton['I-MCP'], dskeleton['I-IP'])
    measures['F-II-PIP'] = angle(dskeleton['II-MCP'], dskeleton['II-PIP'], dskeleton['II-DIP'])
    measures['F-III-PIP'] = angle(dskeleton['III-MCP'], dskeleton['III-PIP'], dskeleton['III-DIP'])
    measures['F-IV-PIP'] = angle(dskeleton['IV-MCP'], dskeleton['IV-PIP'], dskeleton['IV-DIP'])
    measures['F-V-PIP'] = angle(dskeleton['V-MCP'], dskeleton['V-PIP'], dskeleton['V-DIP'])
    measures['F-I-IP'] = angle(dskeleton['I-MCP'], dskeleton['I-IP'], dskeleton['I-TIP'])
    measures['F-II-DIP'] = angle(dskeleton['II-PIP'], dskeleton['II-DIP'], dskeleton['II-TIP'])
    measures['F-III-DIP'] = angle(dskeleton['III-PIP'], dskeleton['III-DIP'], dskeleton['III-TIP'])
    measures['F-IV-DIP'] = angle(dskeleton['IV-PIP'], dskeleton['IV-DIP'], dskeleton['IV-TIP'])
    measures['F-V-DIP'] = angle(dskeleton['V-PIP'], dskeleton['V-DIP'], dskeleton['V-TIP'])
    dskeleton['AXIS'] = dskeleton['II-MCP'] - dskeleton['V-MCP']
    dskeleton['II-P1'] = proj(dskeleton['II-PIP'] - dskeleton['II-MCP'], dskeleton['AXIS'])
    dskeleton['III-P1'] = proj(dskeleton['III-PIP'] - dskeleton['III-MCP'], dskeleton['AXIS'])
    dskeleton['IV-P1'] = proj(dskeleton['IV-PIP'] - dskeleton['IV-MCP'], dskeleton['AXIS'])
    dskeleton['V-P1'] = proj(dskeleton['V-PIP'] - dskeleton['V-MCP'], dskeleton['AXIS'])
    dskeleton['II-P2'] = proj(dskeleton['II-MCP'] - dskeleton['W'], dskeleton['AXIS'])
    dskeleton['III-P2'] = proj(dskeleton['III-MCP'] - dskeleton['W'], dskeleton['AXIS'])
    dskeleton['IV-P2'] = proj(dskeleton['IV-MCP'] - dskeleton['W'], dskeleton['AXIS'])
    dskeleton['V-P2'] = proj(dskeleton['V-MCP'] - dskeleton['W'], dskeleton['AXIS'])
    
    # measures['F-II-MCP'] = angle_between(dskeleton['II-P1'], dskeleton['II-P2'])
    # measures['F-III-MCP'] = angle_between(dskeleton['III-P1'], dskeleton['III-P2'])
    # measures['F-IV-MCP'] = angle_between(dskeleton['IV-P1'], dskeleton['IV-P2'])
    # measures['F-V-MCP'] = angle_between(dskeleton['V-P1'], dskeleton['V-P2'])

    # measures['F-I-MCP'] = angle(dskeleton['I-CMC'], dskeleton['I-MCP'], dskeleton['I-IP'])
    # measures['F-II-PIP'] = angle(dskeleton['II-MCP'], dskeleton['II-PIP'], dskeleton['II-DIP'])
    # measures['F-III-PIP'] = angle(dskeleton['III-MCP'], dskeleton['III-PIP'], dskeleton['III-DIP'])
    # measures['F-IV-PIP'] = angle(dskeleton['IV-MCP'], dskeleton['IV-PIP'], dskeleton['IV-DIP'])
    # measures['F-V-PIP'] = angle(dskeleton['V-MCP'], dskeleton['V-PIP'], dskeleton['V-DIP'])
    # measures['F-I-IP'] = angle(dskeleton['I-MCP'], dskeleton['I-IP'], dskeleton['I-TIP'])
    # measures['F-II-DIP'] = angle(dskeleton['II-PIP'], dskeleton['II-DIP'], dskeleton['II-TIP'])
    # measures['F-III-DIP'] = angle(dskeleton['III-PIP'], dskeleton['III-DIP'], dskeleton['III-TIP'])
    # measures['F-IV-DIP'] = angle(dskeleton['IV-PIP'], dskeleton['IV-DIP'], dskeleton['IV-TIP'])
    # measures['F-V-DIP'] = angle(dskeleton['V-PIP'], dskeleton['V-DIP'], dskeleton['V-TIP'])

    measures['OP-I-MCP'] = angle(dskeleton['I-IP'], midpoint(dskeleton['II-MCP'], dskeleton['I-MCP']), dskeleton['II-PIP'], -1)
    measures['OP-II-MCP'] = angle(dskeleton['II-PIP'], midpoint(dskeleton['III-MCP'], dskeleton['II-MCP']), dskeleton['III-PIP'], -1)
    measures['OP-III-MCP'] = angle(dskeleton['III-PIP'], midpoint(dskeleton['IV-MCP'], dskeleton['III-MCP']), dskeleton['IV-PIP'], -1)
    measures['OP-IV-MCP'] = angle(dskeleton['IV-PIP'], midpoint(dskeleton['V-MCP'], dskeleton['IV-MCP']), dskeleton['V-PIP'], -1)
    # orig = dict()#np.ndarray(shape=(4,3))
    # dest = dict()#np.ndarray(shape=(4,3))
    
    dskeleton['II-MC'] = dskeleton['II-MCP'] - dskeleton['W']
    dskeleton['III-MC'] = dskeleton['III-MCP'] - dskeleton['W']
    dskeleton['IV-MC'] = dskeleton['IV-MCP'] - dskeleton['W']
    dskeleton['V-MC'] = dskeleton['V-MCP'] - dskeleton['W']
    dskeleton['N'] = np.cross(dskeleton['II-MC']/np.linalg.norm(dskeleton['II-MC']), dskeleton['V-MC']/np.linalg.norm(dskeleton['V-MC'])) #normal of plane containing W, II-MCP and V-MCP
    # dskeleton['N'] = dskeleton['N']/np.linalg.norm(dskeleton['N'])
    dskeleton['II-P'] = proj(dskeleton['II-PIP'] - dskeleton['II-MCP'], dskeleton['N'])
    dskeleton['III-P'] = proj(dskeleton['III-PIP'] - dskeleton['III-MCP'], dskeleton['N'])
    dskeleton['IV-P'] = proj(dskeleton['IV-PIP'] - dskeleton['IV-MCP'], dskeleton['N'])
    dskeleton['V-P'] = proj(dskeleton['V-PIP'] - dskeleton['V-MCP'], dskeleton['N'])
    #measures['A-II-MCP'] = signed_angle_between(dskeleton['II-MC'], dskeleton['II-P'], dskeleton['N'])
    #measures['A-III-MCP'] = signed_angle_between(dskeleton['III-MC'], dskeleton['III-P'], dskeleton['N'])
    #measures['A-IV-MCP'] = signed_angle_between(dskeleton['IV-MC'], dskeleton['IV-P'], dskeleton['N'])
    #measures['A-V-MCP'] = signed_angle_between(dskeleton['V-MC'], dskeleton['V-P'], dskeleton['N'])
    measures['A-II-TIP'] = np.linalg.norm(dskeleton['I-TIP'] - dskeleton['II-TIP'], ord=2)
    measures['A-III-TIP'] = np.linalg.norm(dskeleton['II-TIP'] - dskeleton['III-TIP'], ord=2)
    measures['A-IV-TIP'] = np.linalg.norm(dskeleton['III-TIP'] - dskeleton['IV-TIP'], ord=2)
    measures['A-V-TIP'] = np.linalg.norm(dskeleton['IV-TIP'] - dskeleton['V-TIP'], ord=2)
    
    # print('N: ', dskeleton['N'])
    # print('V-MC: ', dskeleton['V-MC'])
    # print('V-PIP: ', dskeleton['V-PIP'] - dskeleton['V-MCP'])
    # print('V-P: ', dskeleton['V-P'])
    # cross = np.cross(dskeleton['V-MC'], dskeleton['V-P'])
    # print(unit_vector(cross))
    # print(np.dot(dskeleton['N'], cross))
    # cross = np.cross(v1_u, v2_u)
    # if (np.dot(normal, cross) < 0):
    
    # for measure in measures:
    #     print(measure, measures[measure])
    # angles = np.ndarray(shape=(4))
    # orig[0] = skeleton[2] - skeleton[0]
    # orig[1] = skeleton[3] - skeleton[0]
    # orig[2] = skeleton[4] - skeleton[0]
    # orig[3] = skeleton[5] - skeleton[0]
    # n = np.cross(orig[0], orig[3])
    # n = n/np.linalg.norm(n)
    # dest[0] = proj(skeleton[11] - skeleton[2], n)
    # dest[1] = proj(skeleton[14] - skeleton[3], n)
    # dest[2] = proj(skeleton[17] - skeleton[4], n)
    # dest[3] = proj(skeleton[20] - skeleton[5], n)
    # angles[0] = angle_between(orig[0], dest[0])
    # angles[1] = angle_between(orig[1], dest[1])
    # angles[2] = angle_between(orig[2], dest[2])
    # angles[3] = angle_between(orig[3], dest[3])
    
    positions['F-I-CMC'] = pskeleton['I-CMC']
    positions['F-II-MCP'] = pskeleton['II-MCP']
    positions['F-III-MCP'] = pskeleton['III-MCP']
    positions['F-IV-MCP'] = pskeleton['IV-MCP']
    positions['F-V-MCP'] = pskeleton['V-MCP']
    positions['F-I-MCP'] = pskeleton['I-MCP']
    positions['F-II-PIP'] = pskeleton['II-PIP']
    positions['F-III-PIP'] = pskeleton['III-PIP']
    positions['F-IV-PIP'] = pskeleton['IV-PIP']
    positions['F-V-PIP'] = pskeleton['V-PIP']
    positions['F-I-IP'] = pskeleton['I-IP']
    positions['F-II-DIP'] = pskeleton['II-DIP']
    positions['F-III-DIP'] = pskeleton['III-DIP']
    positions['F-IV-DIP'] = pskeleton['IV-DIP']
    positions['F-V-DIP'] = pskeleton['V-DIP']
    #positions['A-II-MCP'] = pskeleton['II-MCP']
    #positions['A-III-MCP'] = pskeleton['III-MCP']
    #positions['A-IV-MCP'] = pskeleton['IV-MCP']
    #positions['A-V-MCP'] = pskeleton['V-MCP']
    positions['A-II-TIP'] = pskeleton['II-TIP']
    positions['A-III-TIP'] = pskeleton['III-TIP']
    positions['A-IV-TIP'] = pskeleton['IV-TIP']
    positions['A-V-TIP'] = pskeleton['V-TIP']
    positions['OP-I-MCP'] = midpoint(dskeleton['II-MCP'], dskeleton['I-MCP'])
    positions['OP-II-MCP'] = midpoint(dskeleton['III-MCP'], dskeleton['II-MCP'])
    positions['OP-III-MCP'] = midpoint(dskeleton['IV-MCP'], dskeleton['III-MCP'])
    positions['OP-IV-MCP'] = midpoint(dskeleton['V-MCP'], dskeleton['IV-MCP'])
    # for m in measures:
    #     print m, measures[m]
    return measures, positions

    


# def abd_measures(skeleton):
#     orig = np.ndarray(shape=(4,3))
#     dest = np.ndarray(shape=(4,3))
#     angles = np.ndarray(shape=(4))
#     orig[0] = skeleton[2] - skeleton[0]
#     orig[1] = skeleton[3] - skeleton[0]
#     orig[2] = skeleton[4] - skeleton[0]
#     orig[3] = skeleton[5] - skeleton[0]
#     n = np.cross(orig[0], orig[3])
#     n = n/np.linalg.norm(n)
#     dest[0] = proj(skeleton[11] - skeleton[2], n)
#     dest[1] = proj(skeleton[14] - skeleton[3], n)
#     dest[2] = proj(skeleton[17] - skeleton[4], n)
#     dest[3] = proj(skeleton[20] - skeleton[5], n)
#     angles[0] = angle_between(orig[0], dest[0])
#     angles[1] = angle_between(orig[1], dest[1])
#     angles[2] = angle_between(orig[2], dest[2])
#     angles[3] = angle_between(orig[3], dest[3])
#     # print(orig)
#     # print(dest)
#     # print(n)
#     # print(angles)
#     return angles

def graphicAngles(wp, dataset = 'hands17', smooth=11):
    import pyrealsense2 as rs
    from sensor_utils import intrinsics

    if dataset == 'icvl':
        njoints = 16
    elif dataset == 'nyu':
        njoints = 14
    elif dataset == 'msra':
        njoints = 21        
    elif dataset == 'hands17':
        njoints = 21        

    y = np.ndarray(shape=(njoints,3))
    
    for i in range(njoints):
        for j in range(3):
            wp[:,i,j] = signal.savgol_filter(wp[:,i,j], int(smooth), 3)
    
    angle = np.ndarray(shape=(wp.shape[0], len(nameangles)), dtype=float)
    positions = np.ndarray(shape=(wp.shape[0], len(nameangles), 3), dtype=float)
    intr = intrinsics()
    for i in range(wp.shape[0]):
        x, y = skeletonMeasures(wp[i], dataset, intr)
        for item in range(len(x)):
            angle[i,item] = x[nameangles[item]]
            positions[i,item, 0] = y[nameangles[item]][0]
            positions[i,item, 1] = y[nameangles[item]][1]
            positions[i,item, 2] = y[nameangles[item]][2]
            
    fig2 = plt.figure(figsize=(25, 20))
    fig2.tight_layout()

    spec2 = gridspec.GridSpec(nrows=3, ncols=2+(len(nameangles))/3)


    for i in range(len(nameangles)):
        # print(len(angle[:][i]))
        print(i, i%3, i/3, len(nameangles))
        f = fig2.add_subplot(spec2[i%3, i/3])
        f.set_title(nameangles[i])
        f.plot(angle[:,i], label=str(i))
    
    return fig2

def drawAngles(img, joints, dataset, angle, intr): 
    import pyrealsense2 as rs
    from sensor_utils import intrinsics
   
    if intr == None:
        intr = intrinsics()
    img = show_results(img, joints, dataset)#util.draw_pose(args.dataset, img, joints)
    angles, positions = skeletonMeasures(joints, dataset, intr)
    
    if angle == -1:
        for point in nameangles:
            text = str("%.1f" % angles[point])
            # print(text, (int(positions[point,0]),int(positions[point,1])))
            cv2.putText(img, text, (int(positions[point][0]),int(positions[point][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),2,cv2.LINE_AA)
    else:
        text = str("%.1f" % angles[angle])
        # print(text, (int(positions[point,0]),int(positions[point,1])))
        cv2.putText(img, text, (int(positions[angle][0]),int(positions[angle][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),2,cv2.LINE_AA)
    #print(text)
    return img


def show_results(img, results, dataset):
    img = np.minimum(img, 1500)
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(img*255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_show = util.draw_pose(dataset, img, results)
    return img_show


def segmentImage(src, upper_ = 650):       
    seg = cv2.inRange(src, 0, upper_)
    comps = cv2.connectedComponentsWithStats(seg, 4, cv2.CV_32S) #pegar componentes conexas
    l = list()
    for x in comps[2]:
        l.append(x[cv2.CC_STAT_AREA])    
    cc = filter(lambda x: x[cv2.CC_STAT_AREA] > 100 and x[cv2.CC_STAT_WIDTH] < 600, comps[2])
    if (len(cc) > 0):
        i = l.index(cc[0][4])
        src[np.where(comps[1] != i)] = np.max(src)
    #sorted(cc, )
    #if (len(l) > 2):
    #    i = l.index(sorted(l)[-2]) #segunda maior componente em area
    #else:
    #    i = l.index(sorted(l)[-1]) #segunda maior componente em area
    
    return src
