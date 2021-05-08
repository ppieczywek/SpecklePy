import numpy as np
import cv2

buffer_size = 10


def nothing(x):
    pass

cv2.namedWindow('FUJII_algorithm_demo')
cv2.createTrackbar('FUJII_SCALE','FUJII_algorithm_demo',20,100,nothing)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

fujii_buffer = []
total_fujii = np.zeros((height,width), np.float32)

for i in range(buffer_size):
    
    ret, current_frame = cap.read()
    if channels > 1:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    current_frame = current_frame.astype(np.float32) * (1.0 / 255.0)

    abs_diff = cv2.absdiff(current_frame, previous_frame)
    sum = current_frame + previous_frame
    sum += (1.0 / 255.0)

    fujii = cv2.multiply(abs_diff, cv2.pow(sum, -1.0))
    
    fujii_buffer.append(fujii.copy())
    total_fujii += fujii

    previous_frame = current_frame.copy()

last_frame = buffer_size-1

while(True):
    
    my_val = cv2.getTrackbarPos('FUJII_SCALE','FUJII_algorithm_demo')
    max_fujii = 5.0 * ((my_val + 1.0) / 100.0)
    scale_coeff = (1.0 / max_fujii) * 255.0

    ret, current_frame = cap.read()

    if np.shape(current_frame) != ():

        if channels > 1:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        current_frame = current_frame.astype(np.float32) * (1.0 / 255.0)

        abs_diff = cv2.absdiff(current_frame, previous_frame)
        sum = current_frame + previous_frame 
        sum += (1.0 / 255.0)

        fujii = cv2.multiply(abs_diff, cv2.pow(sum, -1.0))
    
        total_fujii -= fujii_buffer[last_frame]
        total_fujii += fujii

        fujii_buffer[last_frame] = fujii.copy()
        
        last_frame += 1

        if last_frame == buffer_size:
            last_frame = 0

        previous_frame = current_frame.copy()

        final =  total_fujii * scale_coeff
        ret, final = cv2.threshold(final, 255, 255,cv2.THRESH_TRUNC)

        final = final.astype(np.uint8)
        final = cv2.GaussianBlur(final,(5,5),0)

        im_color = cv2.applyColorMap(final, cv2.COLORMAP_JET)
        cv2.imshow('FUJII_algorithm_demo', im_color)
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        cap.release()


cv2.waitKey(0)
cv2.destroyAllWindows()

