#!/bin/bash

resultsFolder='data/09-2018/results/original/hands17/skeletons/'
datasets=(hands17)
folders=(original)


for i in `ls data/09-2018/results/original/hands17/angles/*.txt`; do
    [ -f "$i" ] || break    
    python2.7 src/demo/compare-angles.py $i    
done
