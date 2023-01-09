import numpy as np
import cv2
import wx

app = wx.App()
frame = wx.Frame(None, -1, 'video_to_lasca.py')
frame.SetSize(0, 0, 200, 50)

openFileDialog = wx.FileDialog(frame, "Open", "", "",
                               "Uncompressed video data files (*.avi)|*.avi",
                               wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)  #

openFileDialog.ShowModal()
file_list = openFileDialog.GetPaths()
openFileDialog.Destroy()

filter_size = 7
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size)).astype(np.float32)
kernel /= np.sum(kernel)

gray_threshold = 20
max_value = 0.65

save_video = True

if __name__ == '__main__':

    for file in file_list:
        file_name = file.split('\\')[-1]
        print(file_name[:-4])

        cap = cv2.VideoCapture(file)
        if cap.isOpened() is False:
            print("Error opening video stream or file")
            break

        LASCA_VAL = 0
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_depth = int(cap.get(cv2.CAP_PROP_CHANNEL))
        frame_count = 0

        if save_video is True:
            output_video = file_name[:-4] + "_LASCA.avi"

            out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  30, (frame_width, frame_height))

        buffer = np.zeros((frame_height, frame_width), dtype=np.uint8, order='C')
        while(cap.isOpened()):
            ret, current_frame = cap.read()
            if ret is True:

                if len(current_frame.shape) > 2:
                    buffer = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                    frame_count += 1
                else:
                    buffer = current_frame.copy()
                    frame_count += 1

                img = buffer.astype(np.float32, copy=False)
                mean_img = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel)
                mean_img_sq = np.multiply(mean_img, mean_img)
                sq = np.multiply(img, img)
                sq_img_mean = cv2.filter2D(sq, ddepth=cv2.CV_32F, kernel=kernel)
                std = cv2.subtract(sq_img_mean, mean_img_sq)
                std[std <= 0] = 0.000001
                cv2.sqrt(std, dst=std)
                mask = mean_img < gray_threshold
                cv2.pow(mean_img, power=-1.0, dst=mean_img)

                LASCA = cv2.multiply(std, mean_img)
                LASCA -= 0.0
                LASCA /= max_value

                LASCA[LASCA < 0.0] = 0.0
                LASCA[LASCA > 1.0] = 1.0

                LASCA = 1 - LASCA
                LASCA = cv2.GaussianBlur(LASCA, (5, 5), 0)
                LASCA[mask] = 0
                LASCA *= 255
                LASCA = LASCA.astype(np.uint8)
                cv2.filter2D(LASCA, dst=LASCA, ddepth=cv2.CV_8U, kernel=kernel)

                im_color = cv2.applyColorMap(LASCA, colormap=cv2.COLORMAP_JET)

                if save_video is True:
                    out.write(im_color)

                frame_count += 1
            else:
                break

        cap.release()
        if save_video is True:
            out.release()
