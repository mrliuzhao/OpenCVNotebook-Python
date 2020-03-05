import numpy as np
import cv2
import os
import time
from matplotlib import pyplot as plt


def update(val=0):
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

    start = time.time()
    print('computing disparity...')
    disp = stereo.compute(imgL, imgR)
    disparity = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('disparity1', disparity)

    print(disp.shape)
    disp = disp.astype(np.float32) / 16.0
    plt.hist(disp)
    plt.show()
    disp = (disp - min_disp) / num_disp
    print('计算视差用时：', (time.time() - start))

    cv2.imshow('Origin', cv2.hconcat((imgL, imgR)))
    cv2.imshow('disparity2', disp)


    plt.hist(disp)
    plt.show()


if __name__ == "__main__":
    min_disp = 1
    max_disp = 81
    num_disp = max_disp - min_disp
    blockSize = 5
    uniquenessRatio = 8
    speckleRange = 2
    speckleWindowSize = 64
    disp12MaxDiff = 1
    # P1 = 600
    # P2 = 2400
    P1 = 8 * 1 * blockSize * blockSize
    P2 = 32 * 1 * blockSize * blockSize
    path = os.path.abspath('.')
    path1 = os.path.join(path, 'resources', 'left-3.jpg')
    path2 = os.path.join(path, 'resources', 'right-3.jpg')
    print(path1)
    print(path2)
    imgL = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    # imgL = cv2.imread(path1)
    # imgR = cv2.imread(path2)
    for i in range(3):
        imgL = cv2.pyrDown(imgL)
        imgR = cv2.pyrDown(imgR)
    # imgL = cv2.imread(r".\resources\left-2.jpg")
    # imgR = cv2.imread(r".\resources\right-2.jpg")
    print(imgL.shape)
    print(imgR.shape)
    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
    cv2.createTrackbar('blockSize', 'disparity', blockSize, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=blockSize,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )
    update()
    cv2.waitKey()
