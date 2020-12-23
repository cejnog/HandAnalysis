#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from dataset_utils import patients, control
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import tableFormats

subjects = patients + control
path = "/home/cejnog/Documents/Pose-REN/data/07-2019/leave-one-out/svm-f/"
sigmas = ["s1/", "s2/", "s3/", "s4/", "s5/"]
hands = ['l', 'r']
#sigmas = ['s1/']
#hands = ['r']
trainsizes = [100, 200, 300, 400]
maxInstances = 1
#scores = pd.DataFrame(columns=["subject", "trainsize", "hand", "sigma", "score"])
# text = "\\begin{table}[tb] \\caption{SVM Scores.} \\label{exp-summary}\n"
# text += "\\begin{tabular}{|l|r|r|r|r|r|} \\hline\n"
# text += "\\textbf{Subject} & 100 & 200 & 300 & 400 & 500 \\\ \hline\n"

totalscores = dict()
for sigma in sigmas:
    totalscores[sigma] = dict()
    for hand in hands:
        totalscores[sigma][hand] = dict()
        for subject in subjects:
            totalscores[sigma][hand][subject] = np.ndarray((len(trainsizes) + 1, maxInstances))
        for instance in range(0, maxInstances):
            text = PrettyTable(padding_width = 3, header_style = 'upper')
            #text.field_names = ["Patient"] + [p for p in subjects] + ["Average"]#
            text.field_names = ["Patient", "ts=100", "ts=200", "ts=300", "ts=400", "Average"]
            averages = list()
            for subject in subjects:
                scores = list()
                
                fscores = list()
                scores.append(subject)
                k = 0
                for trainsize in trainsizes:
                    if maxInstances > 1:
                        filepath = path + sigma + subject + "_" + hand + "_" + str(trainsize) + "_" + str(instance) + ".txt"
                    else:
                        filepath = path + sigma + subject + "_" + hand + "_" + str(trainsize) + ".txt"
                    
                    if os.path.isfile(filepath):
                        with open(filepath) as f:
                            lines = f.readlines()
                            if len(lines) == 0:
                                scores.append(0)
                            else:
                                score = float(lines[-1])
                                totalscores[sigma][hand][subject][k][instance] = score
                                scores.append("{:3.4f}".format(score))
                                fscores.append(score)
                    else:
                        scores.append(0)
                    k += 1
                scores.append("{:3.4f}".format(np.mean(np.asarray(fscores))))
                totalscores[sigma][hand][subject][k][instance] = np.mean(np.asarray(fscores))
                averages.append(np.mean(np.asarray(fscores)))
                text.add_row(scores)
            print(text.get_string(title="sigma = " + sigma + " hand = " + hand + " instance " + str(instance)))
starts = ['C', 'P']
scores = list()


for hand in hands:
    text = PrettyTable(padding_width = 2, header_style = 'upper')
    text.field_names = ["Patient", "ts=100 C", "ts=100 P", "ts=200 C", "ts=200 P", "ts=300 C", "ts=300 P", "ts=400 C", "ts=400 P"]
    
    for sigma in sigmas:
        fscores = list()
        fscores.append("$\sigma = " + sigma[-2] + "$")

        for k in range(len(trainsizes)): #para cada k
            for letter in starts: #controle ou paciente
                scores = list()
                lsub = [x for x in subjects if x.startswith(letter)] #listo todos os elementos
                for subject in lsub: #pego todos os scores do conjunto
                    score = (np.mean(totalscores[sigma][hand][subject][k][:]))
                    stdev = (np.std(totalscores[sigma][hand][subject][k][:]))
                    scores.append(score)
                    #print(lsub, k, scores)
                fscores.append("${:2.1f}\% \pm {:1.1f}\%$".format(100.0*np.mean(scores), 100.0*np.std(scores)))
                #print(fscores)
        text.add_row(fscores)
    print(text.get_string(title="sigma = " + sigma + " hand = " + hand + " averages"))
    print(tableFormats.latexTable(text))
    