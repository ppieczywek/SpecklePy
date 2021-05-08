# SpecklePy

## Introduction

Repository contains simple Python scripts, designed to demonstrate two techniques of laser speckle imaging. Scrips are based on OpenCV library. The general principle of the programs is to connect to the selected imaging device and start imaging the speckle activity in the live mode.

FUJII.py - implementation of conventional Fujii algorithm
ESF.py - implementation of exponentially smoothed Fujii algorithm

In brief, conventional Fujii coefficient is calculated from a arbitrary chosen number of captured video frames. Therefore, it requires to store a buffer of previously calculated values, which is updated by new data.

The ESF algorithm does not require to store captured frames in separate buffer. The previously calculated values are taken into account by means of  exponentially moving average algorithm. The data accumulation rate can be freely adjusted. High accumulation rate means that only recent data is taken into account.

Please check the following publication for more detailed description of both methods:
Pieczywek et al. (2017), "Exponentially smoothed Fujii index for online imaging of biospeckle spatial activity" Computers and Electronics in Agriculture
142, pp. 70-78
https://doi.org/10.1016/j.compag.2017.08.018


## General description

Required Python packages (as detailed in "requirements.txt"):
* opencv-python
* numpy

ESF script runs in two separate windows - control window and preview window. Preview window shows processed or unprocessed video image. Control window contains trackbars used to manipulate the data processing input parameters. FUJII script runs in single window with SCALE trackbar only.

Trackbars description:

* MIN_GRAY  - pixel gray level threshold; pixels with gray level lower than MIN_GRAY will not be analyzed (only for ESF.py)

* MIN_DIFF  - pixels minimum gray level difference between two consecutive frames; pixels which show changes in gray level lower than MIN_DIFF will not be analyzed (only for ESF.py)
* SCALE - maximal value of FUJII/ESF activity level; used to adjust color scale of output data (available for both ESF.py and FUJII.py)
* ACC_RATE - available only in case of ESF algorithm; adjust the frame accumulation rate of the exponentially moving average algorithm (only for ESF.py)
* OFF/ON - on/off switch between raw image and live processing mode (only for ESF.py)

## Script usage

Pressing "q" on keyboard terminates both scripts.

Switching OFF/ON slider to position "1" turns on the live processing mode (for ESF.py).

**Before running the script set the correct value of the "device_id" variable. Setting this value to "0" means "connect with the first capturing device".  If you are using laptop with build-in webcam, then "0" refers to this camera. Any other device, such as USB external camera will have higher id numbers (1, 2 etc.).**

In case of FUJII.py you can adjust the size of frame buffer using "buffer_size" variable. The default value is 10 frames.

Both scripts connect with capturing devices using their default settings. If you want to adjust the video acquisition parameters put the additional lines of code.

