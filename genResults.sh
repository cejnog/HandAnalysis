#!/bin/bash

resultsFolder='data/09-2018/results'
datasets=(hands17)
#folders=(original segmentation)
folders=(original)


for i in `ls data/09-2018/bags/C2_*.bag`; do
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
            if [ $k = "segmentation" ];
            then
                echo python src/demo/realsense-bag-tracker.py -i $i -d $j -hand $hand -o ${resultsFolder}/${k}/${j}/videos/${w}.avi -j ${resultsFolder}/${k}/${j}/skeletons/${w}.txt -s True
            else 
                echo python src/demo/realsense-bag-tracker.py -i $i -d $j -hand $hand -o ${resultsFolder}/${k}/${j}/videos/${w}.avi -j ${resultsFolder}/${k}/${j}/skeletons/${w}.txt
            fi
        # do whatever on $i
        done
    # do whatever on $i
    done
    
done
