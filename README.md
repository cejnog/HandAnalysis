# HandAnalysis
Code implementation for the paper Hand pose estimation of patients with rheumatoid arthritis

> [Project page](http://vision.ime.usp.br/~cejnog/handanalysis/)

> [Dataset page](http://vision.ime.usp.br/~cejnog/handanalysis/dataset/)

Requirements:
* python2.7
* [Pose-REN demo](https://github.com/xinghaochen/Pose-REN) - Available with the script init.sh
* Requirement libs: 
  - `xarray`
  - `opencv-python`
  - `tqdm`
  - `logging`
  - `PIL`
  - `argparse`
  - `pyrealsense2`

Usage:
1. `chmod +x init.sh`
2. `./init.sh`
3. `pip2.7 install -r requirements.txt`
4. `python2.7 src/demo/rs_tracker_ppm.py (ppmfolder)`
