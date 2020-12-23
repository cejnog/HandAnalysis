#!/bin/bash

resultsFolder='data/09-2018/results'
datasets=(nyu icvl msra hands17)
folders=(original segmentation)


for i in `ls data/09-2018/bags/*.bag`; do
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
	    python src/demo/plot.py -d $j -f ${resultsFolder}/${k}/${j}/skeletons/${w}.txt -o ${resultsFolder}/${k}/${j}/skeletons/angle_${w}.png
        # do whatever on $i
        done
    # do whatever on $i
    done
    
done
