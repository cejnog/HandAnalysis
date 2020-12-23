import multiprocessing as mp
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.svm import LinearSVC, SVC
from skrvm import RVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from skeleton_utils import read_angles_from_file, read_skeleton_from_file
import random
import dataset3_utils
from result_comparison import subresolutionsized
from graphic_utils import skeletonMeasures, nameangles
import numpy as np
import os, sys, shutil
from tqdm import tqdm
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.neighbors import KNeighborsClassifier

global X_train, X_test, Y_train, Y_test
global FX_train, FX_test, FY_train, FY_test

def addSampleTrain(k):
    global X_train, Y_train
    angle = read_angles_from_file(FX_train[k])
    key = FX_train[k].split("/")[-1].split("_")[0] + "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0]
    
    print(key, len(X_train))
    for lm in dataset2_utils.landmarks[key]:
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
    for lm in dataset2_utils.landmarks[key]:
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

def __main__():
    global X_train, X_test, Y_train, Y_test
    path = "data/09-2019/results/original/hands17/angles/"
    #sigmas = ["s2/", "s3/", "s4/"]
    hand = ['l', 'r']
    lenTest = 100
    res_size = 100
    usefft = False
    subjects = dataset3_utils.control + dataset3_utils.patients
    results_folder = "data/11-2019/LOO-nofft/"
    #h = 'r'
    #for sigma in sigmas:   
    pfiles = list()
    pclasses = list()
    cfiles = list()
    cclasses = list()
    filesbyuser = dict()
    angles = dict()
    
    pbar = tqdm(range(100 * len(subjects)), dynamic_ncols=True, desc="Reading files")
    
    angleslist = dataset3_utils.control_angles
    for x in dataset3_utils.patient_angles:
        angleslist[x] = dataset3_utils.patient_angles[x]
    for f in sorted(angleslist):
        for h in hand:
            key = f.split("/")[-1].split(".")[0]
            for user in sorted(subjects):
                h2 = f.split("_")[-1]
                #print(f, user)
                if f.startswith(user) and 'f' in f and h in f and key in dataset3_utils.landmarks and user in dataset3_utils.patients:
                    angles[f] = read_angles_from_file(angleslist[f])
                    pfiles.append(f)
                    pclasses.append(int((user in dataset3_utils.patients) == True))   
                if f.startswith(user) and 'f' in f and h in f and key in dataset3_utils.landmarks and not (user in dataset3_utils.patients):
                    cfiles.append(f)
                    cclasses.append(int((user in dataset3_utils.patients) == True))   
                    angles[f] = read_angles_from_file(angleslist[f])    
                pbar.update(1)
    print(subjects)
    #for chosen in ["C4"]:
    users = list()
    for subj in subjects:
        for h in hand:
            users.append(subj + "_f" + h)
    for chosen in users:
        cx = list(zip(cfiles, cclasses))
        px = list(zip(pfiles, pclasses))
        print(chosen)
        random.shuffle(cx)
        random.shuffle(px)
        
        result_file = results_folder + "/" + chosen + ".txt"

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
        print(x1, x2)
        print(x3)
        if len(x3) > 0:
            x = x1 + x2
            FX_train, FY_train = zip(*x)
            FX_test, FY_test = zip(*x3)  
        
            f = open(result_file, 'w+')
            pbar2 = tqdm(range(len(FX_train)), desc="Creating training set")
            print(chosen)
            for k in range(len(FX_train)):
                angle = angles[FX_train[k]]
                #print(FX_train[k].split("/")[-1].split(".")[0])
                #print(FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0])#+ "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0])
                key = FX_train[k].split("/")[-1].split(".")[0]#FX_train[k].split("/")[-1].split("_")[0] + "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0]
                #os.system("clear")
                pbar2.update(1)
                #print(key, len(X_train))
                for lm in dataset3_utils.landmarks[key]:
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
                    X_train.append(vec)
                    Y_train.append(FY_train[k])

            #print(nameangles[i], end="\t", file=f)
            pbar3 = tqdm(range(len(FX_test)), desc="Creating test set")
            for k in range(len(FX_test)):
                angle = angles[FX_test[k]]
                key = FX_test[k].split("/")[-1].split(".")[0]# + "_" + FX_test[k].split("/")[-1].split("_", 2)[2].split(".")[0]
                #os.system("clear")
                pbar3.update(1)
                #print(key, len(X_test))
                for lm in dataset3_utils.landmarks[key]:
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
            print(np.asarray(X_test).shape, np.asarray(X_train).shape)
            #clf = KNeighborsClassifier(n_neighbors=1)
            clf = LinearSVC(random_state=0, tol=1e-5)
            clf = clf.fit(np.asarray(X_train), np.asarray(Y_train))
            Y_pred = clf.predict(np.asarray(X_test))
            print(classification_report(Y_test, Y_pred), end="\n", file=f)
            print(confusion_matrix(Y_test, Y_pred), end="\n", file=f)
            print(clf.score(X_test, Y_test), end="\n", file=f)
            f.close()

__main__()