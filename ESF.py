import numpy as np
import sys
import cv2
import platform
import time


device_id = 1

cv2.namedWindow('Video preview', cv2.WINDOW_NORMAL)
cv2.namedWindow('Control panel')
cv2.createTrackbar('ACC_RATE', 'Control panel', 80, 100, lambda x: x)
cv2.createTrackbar('MIN_GRAY', 'Control panel', 10, 255, lambda x: x)
cv2.createTrackbar('MIN_DIFF', 'Control panel', 1, 10, lambda x: x)
cv2.createTrackbar('SCALE', 'Control panel', 20, 100, lambda x: x)
cv2.createTrackbar('EXPOSURE', 'Control panel', 5, 13, lambda x: x)
cv2.createTrackbar('GAIN', 'Control panel', 0, 740, lambda x: x)
cv2.createTrackbar('OFF/ON', 'Control panel', 0, 1, lambda x: x)

if platform.system() == "Darwin":
    cap = cv2.VideoCapture(device_id)
elif platform.system() == "Windows":
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
else:
    print("Unable to connect with selected capturing device")
    cv2.destroyAllWindows()
    sys.exit(0)


cv2.waitKey(1500)

if cap.isOpened() is False:
    print("Unable to connect with selected capturing device")
    cv2.destroyAllWindows()
    sys.exit(0)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_GAIN, 260)

ret, current_frame = cap.read()

height = 0
width = 0 
channels = 1
if len(current_frame.shape) == 2:
    height, width = current_frame.shape
    channels = 1
else:
    height, width, channels = current_frame.shape
    
if channels > 1:
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

current_frame = current_frame.astype(np.float32) * (1.0 / 255.0)
previous_frame = current_frame.copy()
weighted_esf = np.zeros((height,width,1), np.uint8)
exposure = -5
gain = 0

while(True):

    if cv2.getWindowProperty('Control panel', cv2.WND_PROP_VISIBLE) == 1:
        value = cv2.getTrackbarPos('ACC_RATE', 'Control panel')
        acc_rate = value / 100.0

        value = cv2.getTrackbarPos('MIN_GRAY', 'Control panel')
        min_gray = value / 255.0

        value = cv2.getTrackbarPos('MIN_DIFF', 'Control panel')
        min_diff = value / 255.0

        value = cv2.getTrackbarPos('SCALE', 'Control panel')
        max_esf = 0.25 * ((value + 1.0) / 100.0)
        scale_coeff = (1.0 / max_esf) * 255.0
         
        value = cv2.getTrackbarPos('EXPOSURE', 'Control panel')
        if -value != exposure:
            exposure = -value
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        value = cv2.getTrackbarPos('GAIN', 'Control panel')
        if value != gain:
            gain = value
            cap.set(cv2.CAP_PROP_GAIN, gain+260)

        s = cv2.getTrackbarPos('OFF/ON', 'Control panel')
    else:
        cap.release()
        break

    ret, current_frame = cap.read()

    if np.shape(current_frame) != ():

        if channels > 1:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if s == 1:
            current_frame = current_frame.astype(np.float32) * (1.0/255.0)
            sum = current_frame + previous_frame
            diff = cv2.absdiff(current_frame, previous_frame)
        
            diff[diff < min_diff] = 0
            sum[sum < min_gray] = 1000
        
            esf = cv2.multiply(diff, cv2.pow(sum, -1.0))
            esf *= scale_coeff
            esf[esf > 255] = 255
            esf[esf < 0] = 0

            esf = esf.astype(np.uint8)
            esf = cv2.GaussianBlur(esf,(5,5),0)

            weighted_esf = cv2.addWeighted(weighted_esf, 1.0 - acc_rate, esf, acc_rate, gamma = 0)
            im_color = cv2.applyColorMap(weighted_esf, cv2.COLORMAP_JET)
            cv2.imshow('Video preview', im_color)
         
            previous_frame = current_frame.copy()
        else:
            cv2.imshow('Video preview', current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if cv2.waitKey(20) & 0xFF == ord('s'):
            timestr = time.strftime("%Y%m%d_%H%M%S")
            if s == 1:
                filename = "speckle_map_" + timestr + ".tif"
                cv2.imwrite(filename, im_color)
            else:
                filename = "raw_image_" + timestr + ".tif"
                cv2.imwrite(filename, current_frame)
                
                
    else:
        cap.release()

cap.release()
cv2.destroyAllWindows()

