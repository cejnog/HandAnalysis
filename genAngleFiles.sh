#!/bin/bash

resultsFolder='data/09-2018/results'
datasets=(hands17)
#folders=(original segmentation)
folders=(original)


for i in `ls data/09-2018/results/original/hands17/skeletons/*.txt`; do
    [ -f "$i" ] || break    
    x=${i##*_}
    hand=${x:1:1}
    y=${i##*/}    
    w=${y%.*}
    echo $w
    for j in "${datasets[@]}";
    do
        for k in "${folders[@]}";
        do
            python2.7 src/demo/skeleton2angle.py -s ${resultsFolder}/${k}/${j}/skeletons/${w}.txt -a ${resultsFolder}/${k}/${j}/angles/${w}.txt
        done
    done
    
done
