import multiprocessing as mp
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from skeleton_utils import read_angles_from_file, read_skeleton_from_file
import random
import dataset_utils
from result_comparison import subresolutionsized
from graphic_utils import skeletonMeasures, nameangles
import numpy as np
import os, sys, shutil
from tqdm import tqdm
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

global X_train, X_test, Y_train, Y_test
global FX_train, FX_test, FY_train, FY_test

def addSampleTrain(k):
    global X_train, Y_train
    angle = read_angles_from_file(FX_train[k])
    key = FX_train[k].split("/")[-1].split("_")[0] + "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0]
    
    print(key, len(X_train))
    for lm in dataset_utils.landmarks[key]:
        vec = list()
        for i in range(len(nameangles)):
            s = subresolutionsized(angle[lm[0]:lm[1], i], res_size)
            if usefft:
                coeffs = abs(fft(s))[1:res_size/2 - 1]
                for coef in coeffs:
                    vec.append(coef)
            else:
                for element in s:
                    vec.append(element)
        X_train.append(vec)
        Y_train.append(FY_train[k])

def addSampleTest(k):
    angle = read_angles_from_file(FX_test[k])
    key = FX_train[k].split("/")[-1].split("_")[0] + "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0]
    global X_test, Y_test
    print(key, len(X_test))
    for lm in dataset_utils.landmarks[key]:
        vec = list()
        for i in range(len(nameangles)):
            s = subresolutionsized(angle[lm[0]:lm[1], i], res_size)
            if usefft:
                coeffs = abs(fft(s))[1:res_size/2 - 1]
                for coef in coeffs:
                    vec.append(coef)
            else:
                for element in s:
                    vec.append(element)
        X_test.append(vec)
        Y_test.append(FY_train[k])

def plotCoefs(coefs, filename):
    fig = plt.figure(figsize=(15,12))
    vallist = list()
    ax = plt.axes()
    for x in range(len(nameangles)):
        sum = 0
        for y in range(int(len(coefs) / len(nameangles))):
            sum += np.abs(coefs[len(nameangles)*y + x])
        sum /= len(coefs) / len(nameangles)
        vallist.append(sum)
    n = 4
    widths = np.arange(0, n*len(nameangles), n) 
    plt.bar(widths, vallist, width=3, tick_label=nameangles)

    plt.xticks(rotation='vertical')
    ax.set_ylim(0, 0.0002)
    plt.savefig(filename)

def __main__():
    global X_train, X_test, Y_train, Y_test
    path = "data/07-2019/leave-one-out/angles/"
    sigmas = ["s1/"]
    hand = ['l', 'r']
    lenTest = 100
    lenTrains = [100, 200, 300, 400]
    res_size = 100
    usefft = False
    subjects = dataset_utils.control + dataset_utils.patients
    results_folder = "data/07-2019/leave-one-out/svm-f"
    for h in hand:
        for sigma in sigmas:   
            for lenTrain in lenTrains:
                pfiles = list()
                pclasses = list()
                cfiles = list()
                cclasses = list()
                filesbyuser = dict()
                for f in sorted(os.listdir(path + sigma)):
                    for user in subjects:
                        h2 = f.split("_")[-1]
                        if f.startswith(user) and h in h2 and user in dataset_utils.patients:
                            pfiles.append(path + sigma + f)
                            pclasses.append(int((user in dataset_utils.patients) == True))   
                        if f.startswith(user) and h in h2 and not (user in dataset_utils.patients):
                            cfiles.append(path + sigma + f)
                            cclasses.append(int((user in dataset_utils.patients) == True))           
                chosen = 'C1'
                #for chosen in subjects:
                cx = list(zip(cfiles, cclasses))
                px = list(zip(pfiles, pclasses))

                random.shuffle(cx)
                random.shuffle(px)
                
                result_file = results_folder + "/" + sigma + chosen + "_" + h + "_" + str(lenTrain) + ".txt"
                result_file_weights = results_folder + "/" + str(lenTrain) + "-weights.png"

                X_train = list()
                X_test = list()
                Y_train = list()
                Y_test = list()
                FX_train = list()
                FX_test = list()
                FY_train = list()
                FY_test = list()

                x1 = [f for f in px if not(chosen in f[0])]
                x2 = [f for f in cx if not(chosen in f[0])]
                x3 = [f for f in cx + px if chosen in f[0]]
                x = x1[0:int(lenTrain/2)] + x2[0:int(lenTrain/2)]
                FX_train, FY_train = zip(*x)
                FX_test, FY_test = zip(*x3[0:lenTest])  
                f = open(result_file, 'w')

                pbar = tqdm(range(len(FX_train) + len(FX_test)))
                print(chosen)
                for k in range(len(FX_train)):
                    angle = read_angles_from_file(FX_train[k])
                    key = FX_train[k].split("/")[-1].split("_")[0] + "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0]
                    #os.system("clear")
                    pbar.update(1)
                    for lm in dataset_utils.landmarks[key]:
                        vec = list()
                        #print(key)
                        for i in range(len(nameangles)):
                            s = subresolutionsized(angle[lm[0]:lm[1], i], res_size)
                            if usefft:
                                coeffs = abs(fft(s))[1:int(res_size/2) - 1]
                                for coef in coeffs:
                                    vec.append(coef)
                            else:
                                for element in s:
                                    vec.append(element)
                        X_train.append(vec)
                        Y_train.append(FY_train[k])

                for k in range(len(FX_test)):
                    angle = read_angles_from_file(FX_test[k])
                    key = FX_test[k].split("/")[-1].split("_")[0] + "_" + FX_test[k].split("/")[-1].split("_", 2)[2].split(".")[0]
                    #os.system("clear")
                    pbar.update(1)
                    #print(key, len(X_test))
                    for lm in dataset_utils.landmarks[key]:
                        vec = list()
                        for i in range(len(nameangles)):
                            s = subresolutionsized(angle[lm[0]:lm[1], i], res_size)
                            if usefft:
                                coeffs = abs(fft(s))[1:int(res_size/2) - 1]
                                for coef in coeffs:
                                    vec.append(coef)
                            else:
                                for element in s:
                                    vec.append(element)
                        X_test.append(vec)
                        Y_test.append(FY_test[k])

                clf = LinearSVC(random_state=0, tol=1e-5)
                clf = clf.fit(X_train, Y_train)
                Y_pred = clf.predict(X_test)
                coefs = clf.coef_[0]
                plotCoefs(coefs, result_file_weights)
                f.close()

__main__()