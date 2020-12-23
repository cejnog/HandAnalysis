import multiprocessing as mp
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.svm import LinearSVC
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

global X_train, X_test, Y_train, Y_test
global FX_train, FX_test, FY_train, FY_test

def addSampleTrain(k):
    global X_train, Y_train
    angle = read_angles_from_file(FX_train[k])
    key = FX_train[k].split("/")[-1].split("_")[0] + "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0]
    
    print(key, len(X_train))
    for lm in dataset3_utils.landmarks[key]:
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
    for lm in dataset3_utils.landmarks[key]:
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
    path = "data/11-2019/loo-clean/angles/"
    #sigmas = ["s1/", "s2/", "s4/"]
    sigmas = ["s1/"]
    hand = ['r', 'l']
    lenTest = 100
    lenTrains = [100, 400]
    res_size = 100
    usefft = True
    subjects = dataset3_utils.control + dataset3_utils.patients
    
    results_folder = "data/11-2019/loo-augmentation/svm"
    #sigmas = ['s1/']
    #h = 'r'
    fourier_size = 25
    maxInstances = 7
    users = list()
    for subj in subjects:
        for h in hand:
            users.append(subj + "_f" + h)
    print(users)
    for sigma in sigmas:   
        for instance in range(0, maxInstances):
            for lenTrain in lenTrains:
                pfiles = list()
                pclasses = list()
                cfiles = list()
                cclasses = list()
                filesbyuser = dict()
                #alimenta conjuntos pfiles e cfiles, com os arquivos dos pacientes e do controle respectivamente
                for f in sorted(os.listdir(path + sigma)):
                    for user in sorted(users):
                        h2 = f.split("_")[-3] + "_" + f.split("_")[-2] 
                        
                        if f.startswith(user) and user[:-3] in dataset3_utils.patients:
                            pfiles.append(path + sigma + f)
                            pclasses.append(1)   
                        if f.startswith(user) and user[:-3] in dataset3_utils.control:
                            cfiles.append(path + sigma + f)
                            cclasses.append(0)      
                #escolhe um paciente/controle (users tem paciente/controle por mão)
                for chosen in sorted(users):
                    cx = list(zip(cfiles, cclasses))
                    px = list(zip(pfiles, pclasses))
                    random.shuffle(cx)
                    random.shuffle(px)
                    result_file = results_folder + "/" + sigma + chosen + "_" + h + "_" + str(lenTrain) + "_" + str(instance) + ".txt"

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
                    #print(x1, x2, x3)
                    if len(x3) > 0:
                        x = x1[0:int(lenTrain/2)] + x2[0:int(lenTrain/2)]
                        FX_train, FY_train = zip(*x)
                        FX_test, FY_test = zip(*x3[0:lenTest]) 
                        #print(x1, x2) 
                        f = open(result_file, 'w')
                        pbar = tqdm(range(len(FX_train) + len(FX_test)))
                        print(chosen)
                        #para cada arquivo do conjunto de treinamento, cria uma amostra (um arquivo contém os dados de uma landmark)
                        for k in range(len(FX_train)):
                            angle = read_angles_from_file(FX_train[k])
                            key = FX_train[k].split("/")[-1].split("_")[0] + "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0]
                            pbar.update(1)
                            vec = list()
                            for i in range(len(nameangles)):
                                s = subresolutionsized(angle[:, i], res_size)
                                if usefft:
                                    coeffs = abs(fft(s))[1:int(fourier_size) - 1]
                                    for coef in coeffs:
                                        vec.append(coef)
                                else:
                                    for element in s:
                                        vec.append(element)
                            X_train.append(vec)
                            Y_train.append(FY_train[k]) #as variáveis X_train e Y_train armazenam as amostras de treinamento

                        #para cada arquivo do conjunto de teste, cria uma amostra (um arquivo contém os dados de uma landmark)
                        for k in range(len(FX_test)):
                            angle = read_angles_from_file(FX_test[k])
                            key = FX_test[k].split("/")[-1].split("_")[0] + "_" + FX_test[k].split("/")[-1].split("_", 2)[2].split(".")[0]
                            #os.system("clear")
                            pbar.update(1)
                            #print(key, len(X_test))
                            vec = list()
                            for i in range(len(nameangles)):
                                s = subresolutionsized(angle[:, i], res_size)
                                if usefft:
                                    coeffs = abs(fft(s))[1:int(fourier_size) - 1]
                                    for coef in coeffs:
                                        vec.append(coef)
                                else:
                                    for element in s:
                                        vec.append(element)
                            X_test.append(vec)
                            Y_test.append(FY_test[k]) #as variáveis X_test e Y_test armazenam as amostras de teste

                        #chama o SVC e classifica, gerando a matriz de confusão e os reports
                        clf = LinearSVC(random_state=0, tol=1e-5)
                        clf = clf.fit(X_train, Y_train)
                        Y_pred = clf.predict(X_test)
                        print(classification_report(Y_test, Y_pred), end="\n", file=f)
                        print(confusion_matrix(Y_test, Y_pred), end="\n", file=f)
                        print(clf.score(X_test, Y_test), end="\n", file=f)
                        f.close()

__main__()