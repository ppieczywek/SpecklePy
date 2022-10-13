# SpecklePy

## Introduction

Repository contains simple Python scripts, designed to demonstrate techniques of laser speckle imaging. Scrips are based on OpenCV library. The general principle of the programs is to connect to capturing device and start imaging the speckle activity in the live mode.


## General description

Required Python packages (as detailed in "requirements.txt"):
* opencv-python
* numpy

**Before running the script set the correct value of the "device_id" variable. Setting this value to "0" means "connect with the first capturing device".  If you are using laptop with build-in webcam, then "0" refers to this camera. Any other device, such as USB external camera will have higher id numbers (1, 2 etc.).**

Pressing "q" on keyboard terminates both scripts.
Switching "OFF/ON" slider to position "1" turns on the live processing mode.

Camera settings are controlled by "EXPOSURE" and "GAIN" sliders.

Scripts allow to capture data as still images or videos. Recording starts by pressing the "s" button on keyboard. Recording modes are selected with the "MODE" and "TIME" sliders. Following options are available:

| MODE  | TIME  | Resulting recording mode  |
|---|---|---|
| 0 |0   | Recording sigle still image in TIFF format  |
| 0  | >0  | Recording series of still images in TIFF format, in intervals set by TIME slider. Capturing stars by pressing "s" button. Pressing the "s" button again stops recording. |
| 1  | 0  | Recording single video into AVI file format. Capturing stars by pressing "s" button. Pressing the "s" button again stops recording.  |
| 1  | >0  | Recording single video into AVI file format. Capturing stars by pressing "s" button. Recording stops automatically after number of seconds set on TIME slider.  |
| 2  |  >=0   |  Recording single time-lapse video into AVI file format. Capturing stars by pressing "s" button. Pressing the "s" button again stops recording. Video frames are captured in intervals set by TIME slider (in seconds). |

### lasca_s.py

Spript calculates speckle activity based on spatially resolved laser speckle contrast analysisi algorith (LASCA). Conrast is calculated as ratio of the standard deviation to the mean value of pixels intensities within local processing window. Size of processing window is controlled by "kernel_size" variable.

The lasca_s.py script runs in two separate windows - control window and preview window. Preview window shows processed or unprocessed video stream. Control window contains trackbars used to control the data processing.

Description of script specific trackbars:
* MIN_GRAY - pixel gray level threshold; pixels with gray level lower than MIN_GRAY will not be analyzed
* SCALE - used to scale values of the output speckle activity map

### esf_s.py

In brief, conventional Fujii coefficient is calculated from a arbitrary chosen number of captured video frames. Therefore, it requires to store a buffer of previously calculated values, which is updated by new data.

The ESF algorithm does not require to store captured frames in separate buffer. The previously calculated values are taken into account by means of exponentially moving average algorithm. The data accumulation rate can be freely adjusted. High accumulation rate means that only recent data is taken into account.

Please check the following publication for more detailed description of both methods: Pieczywek et al. (2017), "Exponentially smoothed Fujii index for online imaging of biospeckle spatial activity" Computers and Electronics in Agriculture 142, pp. 70-78 https://doi.org/10.1016/j.compag.2017.08.018

The esf_s.py script runs in two separate windows - control window and preview window. Preview window shows processed or unprocessed video stream. Control window contains trackbars used to control the data processing.

Description of script specific trackbars:

* MIN_GRAY - pixel gray level threshold; pixels with gray level lower than MIN_GRAY will not be analyzed
* MIN_DIFF  - minimum gray level difference between two consecutive frames; pixels which show changes in gray level lower than MIN_DIFF will not be analyzed
* SCALE - used to scale values of the output speckle activity map
* ACC_RATE - adjusts the data accumulation rate
