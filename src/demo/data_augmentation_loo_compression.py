import os
from skeleton_utils import read_angles_from_file, read_skeleton_from_file
import dataset2_utils
from result_comparison import subresolutionsized
from graphic_utils import skeletonMeasures, nameangles
import dataset2_utils
import pandas as pd
import pickle, random
from tqdm import tqdm
from scipy.fftpack import fft, fftfreq, fftshift

def __main__():
    path = "data/10-2019/leave-one-out/angles/"
    results_folder = "/home/cejnog/Documents/Pose-REN/data/10-2019/leave-one-out/svm"
    sigmas = ["s2/", "s3/", "s4/"]
    hand = ['l', 'r']
    lenTest = 100
    lenTrains = [100, 200, 300, 400]
    res_size = 100
    usefft = True
    subjects = dataset2_utils.control + dataset2_utils.patients
    data = pd.DataFrame()
    for subject in subjects:
        maxInstances = 1
        for h in hand:
            for sigma in sigmas:  
                for instance in range(0, maxInstances):
                    for lenTrain in lenTrains:

                        pfiles = list()
                        pclasses = list()
                        cfiles = list()
                        cclasses = list()
                        filesbyuser = dict()
                        for f in sorted(os.listdir(path + sigma)):
                            for user in subjects:
                                h2 = f.split("_")[-1]
                                if f.startswith(user) and h in h2 and user in dataset2_utils.patients:
                                    pfiles.append(path + sigma + f)
                                    pclasses.append(int((user in dataset2_utils.patients) == True))   
                                if f.startswith(user) and h in h2 and not (user in dataset2_utils.patients):
                                    cfiles.append(path + sigma + f)
                                    cclasses.append(int((user in dataset2_utils.patients) == True))           


                        for chosen in subjects:
                            jsonpath = results_folder + "/train_%s_%s_%s_%d_%d.pkl" % (chosen, h, sigma[1], lenTrain, instance)
                            print(jsonpath)
                            with open(jsonpath, 'wb') as finstance:
                                cx = list(zip(cfiles, cclasses))
                                px = list(zip(pfiles, pclasses))

                                random.shuffle(cx)
                                random.shuffle(px)
                                
                                result_file = results_folder + "/" + sigma + chosen + "_" + h + "_" + str(lenTrain) + "_" + str(instance) + ".txt"

                                trainSamples = list()
                                testSamples = list()
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
                                #f = open(result_file, 'w')
                                pbar = tqdm(range(len(FX_train) + len(FX_test)))
                                print(chosen)
                                for k in range(len(FX_train)):
                                    angle = read_angles_from_file(FX_train[k])
                                    key = FX_train[k].split("/")[-1].split("_")[0] + "_" + FX_train[k].split("/")[-1].split("_", 2)[2].split(".")[0]
                                    #os.system("clear")
                                    pbar.update(1)
                                    #print(key, len(X_train))
                                    for lm in dataset2_utils.landmarks[key]:
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
                                        trainSamples.append((vec, FY_train[k]))

                                for k in range(len(FX_test)):
                                    angle = read_angles_from_file(FX_test[k])
                                    key = FX_test[k].split("/")[-1].split("_")[0] + "_" + FX_test[k].split("/")[-1].split("_", 2)[2].split(".")[0]
                                    #os.system("clear")
                                    pbar.update(1)
                                    #print(key, len(X_test))
                                    for lm in dataset2_utils.landmarks[key]:
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
                                        testSamples.append((vec, FY_train[k]))
                                dumplist = dict()
                                dumplist['train'] = trainSamples
                                dumplist['test'] = testSamples
                                pickle.dump(dumplist, finstance)

__main__()