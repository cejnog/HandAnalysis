#!/bin/bash


for i in `ls data/09-2018/bags/*.bag`; do
    [ -f "$i" ] || break
    ARRAY=( ${i//\//" " } )
    python src/demo/bag2vid.py -i $i
done
