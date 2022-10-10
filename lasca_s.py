import numpy as np
import sys
import cv2
import platform
import time


device_id = 1
is_running = False
start_time = 0

font = cv2.FONT_HERSHEY_SIMPLEX
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)).astype(np.float32)
kernel /= np.sum(kernel)

cv2.namedWindow('Video preview', cv2.WINDOW_NORMAL)
cv2.namedWindow('Control panel')
cv2.createTrackbar('MODE', 'Control panel', 0, 2, lambda x: x)
cv2.createTrackbar('TIME', 'Control panel', 0, 60, lambda x: x)
cv2.createTrackbar('MIN_GRAY', 'Control panel', 10, 255, lambda x: x)
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
weighted_esf = np.zeros((height, width, 1), np.uint8)
exposure = -5
gain = 0

gray = np.zeros((height, width), np.uint8)
mean_img_sq = np.zeros((height, width), np.float32)
sq = np.zeros((height, width), np.float32)
img = np.zeros((height, width), np.float32)
mean_img = np.zeros((height, width), np.float32)
sq_img_mean = np.zeros((height, width), np.float32)
std = np.zeros((height, width), np.float32)
LASCA = np.zeros((height, width), np.uint8)
im_color = np.zeros((height, width, 3), np.uint8)
mask = np.zeros((height, width), bool)

output_frame = np.zeros((height, width, 3), np.uint8)
video_capture = None

while(True):

    if cv2.getWindowProperty('Control panel', cv2.WND_PROP_VISIBLE) == 1:
        value = cv2.getTrackbarPos('MODE', 'Control panel')
        capture_mode = value

        value = cv2.getTrackbarPos('TIME', 'Control panel')
        time_interval = value

        value = cv2.getTrackbarPos('MIN_GRAY', 'Control panel')
        min_gray = value  # / 255.0

        value = cv2.getTrackbarPos('SCALE', 'Control panel')
        if value == 0:
            value = 1
        max_contrast = value / 100.0

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
            img = current_frame.astype(np.float32, copy=False)
            cv2.filter2D(img, dst=mean_img, ddepth=cv2.CV_32F, kernel=kernel)
            np.multiply(mean_img, mean_img, out=mean_img_sq)
            np.multiply(img, img, out=sq)
            cv2.filter2D(sq, dst=sq_img_mean, ddepth=cv2.CV_32F, kernel=kernel)
            cv2.subtract(sq_img_mean, mean_img_sq, dst=std)
            cv2.sqrt(std, dst=std)
            mask = mean_img < min_gray
            cv2.pow(mean_img, power=-1.0, dst=mean_img)
            cv2.multiply(std, mean_img, dst=mean_img, scale=255.0/max_contrast, dtype=cv2.CV_32F)
            mean_img[mean_img > 255.0] = 255.0
            LASCA = mean_img.astype(np.uint8)
            LASCA = 255 - LASCA
            LASCA[mask] = 0
            LASCA = cv2.GaussianBlur(LASCA, (5, 5), 0)
            im_color = cv2.applyColorMap(LASCA, cv2.COLORMAP_JET)
            output_frame = im_color.copy()
        else:
            output_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2RGB)

        if capture_mode == 0:
            if s == 0:
                if is_running is True:
                    cv2.putText(output_frame, 'Snapshot (raw) - running sequence', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(output_frame, 'Snapshot (raw) - single image', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                if is_running is True:
                    cv2.putText(output_frame, 'Snapshot (lasca) - running sequence', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(output_frame, 'Snapshot (lasca) - single image', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if capture_mode == 1:
            if s == 1:
                if is_running is True:
                    cv2.putText(output_frame, 'Video (lasca) - capturing', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(output_frame, 'Video (lasca) - idle', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if s == 0:
                if is_running is True:
                    cv2.putText(output_frame, 'Video (raw) - capturing', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(output_frame, 'Video (raw) - idle', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if capture_mode == 2:
            if s == 1:
                if is_running is True:
                    cv2.putText(output_frame, 'Timelapse video (lasca) - capturing', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(output_frame, 'Timelapse video (lasca) - idle', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if s == 0:
                if is_running is True:
                    cv2.putText(output_frame, 'Timelapse video (raw) - capturing', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(output_frame, 'Timelapse video (raw) - idle', (10, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Video preview', output_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            timestr = time.strftime("%Y%m%d_%H%M%S")
            if capture_mode == 0:
                if time_interval > 0:
                    if is_running is True:
                        is_running = False
                    else:
                        is_running = True
                        timestr = time.strftime("%Y%m%d_%H%M%S")
                        if s == 1:
                            filename = "speckle_map_" + timestr + ".tif"
                            cv2.imwrite(filename, im_color)
                        else:
                            filename = "raw_image_" + timestr + ".tif"
                            cv2.imwrite(filename, current_frame)
                        start_time = time.time()
                else:
                    if s == 1:
                        filename = "speckle_map_" + timestr + ".tif"
                        cv2.imwrite(filename, im_color)
                    else:
                        filename = "raw_image_" + timestr + ".tif"
                        cv2.imwrite(filename, current_frame)

            if capture_mode != 0:
                if is_running is True:
                    video_capture.release()
                    is_running = False
                    print('closing video')
                else:
                    filename = "speckle_video_" + timestr + ".avi"
                    video_capture = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))
                    print("creating video file")
                    is_running = True
                    start_time = time.time()

        if is_running is True:
            if capture_mode == 0:
                if (time.time() - start_time) >= time_interval:
                    timestr = time.strftime("%Y%m%d_%H%M%S")
                    if s == 1:
                        filename = "speckle_map_" + timestr + ".tif"
                        cv2.imwrite(filename, im_color)
                    else:
                        filename = "raw_image_" + timestr + ".tif"
                        cv2.imwrite(filename, current_frame)
                    start_time = time.time()

            if capture_mode == 1:
                if time_interval > 0:
                    if (time.time() - start_time) >= time_interval:
                        video_capture.release()
                        print('closing video')
                        is_running = False
                    else:
                        video_capture.write(output_frame)
                else:
                    video_capture.write(output_frame)

            if capture_mode == 2:
                if time_interval > 0:
                    if (time.time() - start_time) >= time_interval:
                        video_capture.write(output_frame)
                        start_time = time.time()

    else:
        cap.release()

cap.release()
cv2.destroyAllWindows()
