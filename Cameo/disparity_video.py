import numpy as np
import cv2
import os
import time
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # 1 right 2 left
    min_disp = 1
    max_disp = 17
    num_disp = max_disp - min_disp
    blockSize = 15
    uniquenessRatio = 8
    speckleRange = 1
    speckleWindowSize = 10
    disp12MaxDiff = 1
    P1 = 8 * 1 * blockSize * blockSize
    P2 = 32 * 1 * blockSize * blockSize
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=blockSize,
                                   uniquenessRatio=uniquenessRatio,
                                   speckleWindowSize=speckleWindowSize,
                                   speckleRange=speckleRange,
                                   disp12MaxDiff=disp12MaxDiff,
                                   P1=P1,
                                   P2=P2,
                                   mode=cv2.STEREO_SGBM_MODE_HH)

    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    while True:
        succ1 = cap1.grab()
        succ2 = cap2.grab()
        if succ1 and succ2:
            _, frame1 = cap1.retrieve()
            _, frame2 = cap2.retrieve()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Display', cv2.hconcat((frame1, frame2)))
            disp = stereo.compute(frame2, frame1)
            disparity = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow('disparity', disparity)

        if cv2.waitKey(1) == 27:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()



