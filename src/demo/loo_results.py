#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from dataset3_utils import patients, control
import numpy as np
import pandas as pd
from prettytable import PrettyTable

subjects = patients + control
path = "/home/cejnog/Documents/Pose-REN/data/11-2019/LOO-nofft/"
totalscores = dict()
text = PrettyTable(padding_width = 3, header_style = 'upper')
text.field_names = ["Patient", "L", "R"]
for subject in subjects:
    #text.field_names = ["Patient"] + [p for p in subjects] + ["Average"]#
    scores = list()
    scores.append(subject)
    for hand in ['l', 'r']:
        key = subject + '_f' + hand
        filepath = path + key + ".txt"
        if os.path.isfile(filepath):
            with open(filepath) as f:
                lines = f.readlines()
                if len(lines) == 0:
                    scores.append(0)
                else:
                    score = float(lines[-1])
                    scores.append("{:3.4f}".format(score))
        else:
            scores.append(0)
    text.add_row(scores)
        
print(text.get_string(title="NO augmentation"))
    
