import numpy as np
import sys, traceback
import cv2


device_id = 0
def nothing(x):
    pass

cv2.namedWindow('ESF_preview')
cv2.namedWindow('Trackbars')
cv2.createTrackbar('ACC_RATE','Trackbars',80,100,nothing)
cv2.createTrackbar('MIN_GRAY','Trackbars',10,255,nothing)
cv2.createTrackbar('MIN_DIFF','Trackbars',1,10,nothing)
cv2.createTrackbar('SCALE','Trackbars',20,100,nothing)
cv2.createTrackbar('OFF/ON', 'Trackbars',0,1,nothing)

cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
cv2.waitKey(1500)

if cap.isOpened() == False:
    print("Unable to connect with selected capturing device")
    cv2.destroyAllWindows()
    sys.exit(0)


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

while(True):

    value = cv2.getTrackbarPos('ACC_RATE','Trackbars')
    acc_rate = value / 100.0
    
    value = cv2.getTrackbarPos('MIN_GRAY','Trackbars')
    min_gray = value / 255.0

    value = cv2.getTrackbarPos('MIN_DIFF','Trackbars')
    min_diff = value / 255.0

    value = cv2.getTrackbarPos('SCALE','Trackbars')
    max_esf = 0.25 * ((value + 1.0) / 100.0)
    scale_coeff = (1.0 / max_esf) * 255.0

    s = cv2.getTrackbarPos('OFF/ON','Trackbars')

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
            im_color = cv2.applyColorMap(weighted_esf, cv2.COLORMAP_PLASMA)
            cv2.imshow('ESF_preview', im_color)
         
            previous_frame = current_frame.copy()
        else:
            cv2.imshow('ESF_preview', current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()

