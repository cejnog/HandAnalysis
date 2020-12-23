#!/bin/bash

resultsFolder='data/09-2019'
datasets=(hands17)
#folders=(original segmentation)
folders=(original)

for i in `ls $resultsFolder/bags`; do
    x=${i##*.}
    hand=${x:1:1}
    y=${i##*/}    
    w=${y%.*}
    if [[ $i == P* ]]; then
        echo "patient_bags['$w'] = '$resultsFolder/bags/$i'"
    fi
    if [[ $i == C* ]]; then
        echo "control_bags['$w'] = '$resultsFolder/bags/$i'"
    fi
done

for i in `ls $resultsFolder/results/original/hands17/skeletons`; do
    x=${i##*.}
    hand=${x:1:1}
    y=${i##*/}    
    w=${y%.*}
    if [[ $i == P* ]]; then
        echo "patient_skeletons['$w'] = '$resultsFolder/original/hands17/skeletons/$i'"
    fi
    if [[ $i == C* ]]; then
        echo "control_skeletons['$w'] = '$resultsFolder/original/hands17/skeletons/$i'"
    fi
done

for i in `ls $resultsFolder/results/original/hands17/angles`; do
    x=${i##*.}
    hand=${x:1:1}
    y=${i##*/}    
    w=${y%.*}
    if [[ $i == P*.txt ]]; then
        echo "patient_angles['$w'] = '$resultsFolder/original/hands17/angles/$i'"
    fi
    if [[ $i == C*.txt ]]; then
        echo "control_angles['$w'] = '$resultsFolder/original/hands17/angles/$i'"
    fi
done

for i in `ls $resultsFolder/results/original/hands17/angles`; do
    x=${i##*.}
    hand=${x:1:1}
    y=${i##*/}    
    w=${y%.*}
    if [[ $i == *.txt ]]; then
        echo "landmarks['$w'] = []"
    fi
done
