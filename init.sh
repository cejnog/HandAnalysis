#!/bin/bash

git clone https://github.com/xinghaochen/Pose-REN.git
mv Pose-REN/* .
mv Pose-REN/src/demo/* src/demo/
mv Pose-REN/src/libs/ src/
mv Pose-REN/src/testing/ src/
mv Pose-REN/src/utils/ src/
mv Pose-REN/src/config.py.example src/
mv Pose-REN/src/__init__.py src/
mv Pose-REN/src/show_result.py src/
rm -rf Pose-REN/
