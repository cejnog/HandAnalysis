#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from dataset3_utils import patients, control
import numpy as np
import pandas as pd
from prettytable import PrettyTable
hand = ['l', 'r']
subjects = patients + control
users = list()
for subj in subjects:
    for h in hand:
        users.append(subj + "_f" + h)
path = "/home/cejnog/Documents/Pose-REN/data/11-2019/loo-clean/svm/"
#sigmas = ["s1/", "s2/", "s3/", "s4/", "s5/"]
#hands = ['l', 'r']
sigmas = ['s1/', 's2/', 's4/']
trainsizes = [100, 400]
maxInstances = 10
#scores = pd.DataFrame(columns=["subject", "trainsize", "hand", "sigma", "score"])
# text = "\\begin{table}[tb] \\caption{SVM Scores.} \\label{exp-summary}\n"
# text += "\\begin{tabular}{|l|r|r|r|r|r|} \\hline\n"
# text += "\\textbf{Subject} & 100 & 200 & 300 & 400 & 500 \\\ \hline\n"
for sigma in sigmas:
        totalscores = dict()
        for subject in users:
            totalscores[subject] = np.ndarray((len(trainsizes) + 1, maxInstances))
        for instance in range(0, maxInstances):
            text = PrettyTable(padding_width = 3, header_style = 'upper')
            #text.field_names = ["Patient"] + [p for p in subjects] + ["Average"]#
            text.field_names = ["Patient", "ts=100", "ts=400", "Average"]
            averages = list()
            
            for subject in users:
                skip = False
                scores = list()
                
                fscores = list()
                scores.append(subject)
                k = 0
                for trainsize in trainsizes:
                    filepath = path + sigma + subject + "_l_" + str(trainsize) + "_" + str(instance) + ".txt"
                    if os.path.isfile(filepath):
                        with open(filepath) as f:
                            lines = f.readlines()
                            if len(lines) == 0:
                                scores.append(0)
                            else:
                                score = float(lines[-1])
                                totalscores[subject][k][instance] = score
                                scores.append("{:3.4f}".format(score))
                                fscores.append(score)
                    else:
                        scores.append(0)
                        skip = True
                    k += 1
                scores.append("{:3.4f}".format(np.mean(np.asarray(fscores))))
                totalscores[subject][k][instance] = np.mean(np.asarray(fscores))
                averages.append(np.mean(np.asarray(fscores)))
                if not skip:
                    text.add_row(scores)
            
            print(text.get_string(title="sigma = " + sigma + " instance " + str(instance)))
        
        scores = list()
        text = PrettyTable(padding_width = 3, header_style = 'upper')
        text.field_names = ["Patient", "ts=100", "ts=400", "Average"]
        for subject in users:
            scores = list()
            scores.append(subject)
            for k in range(len(trainsizes) + 1):
                score = (np.mean(totalscores[subject][k][:]))
                stdev = (np.std(totalscores[subject][k][:]))
                if score < - 100 or score > 100:
                    score = 0
                if stdev < - 100 or stdev  > 100:
                    stdev = 0
                scores.append("{:3.4f} ± {:3.4f}".format(score, stdev))
            text.add_row(scores)
        print(text.get_string(title="sigma = " + sigma + " averages"))
        #print(np.mean(averages))
                    # text += "{:3.4f}".format(score)
                # if trainsize == trainsizes [-1]:
                    # text += "\\\ \hline \n"
                # else:
                    # text += " & "
                    #scores = scores.append({'subject' : subject, 'trainsize': trainsize, 'hand' : hand, 'sigma' : sigma, 'score' : score}, ignore_index=True)
#print(scores) 
# text += "\end{tabular} \end{table}"
