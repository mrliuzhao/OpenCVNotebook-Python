import cv2
import numpy as np
import matplotlib.pyplot as plt


cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
orb = cv2.ORB_create()
bfMat = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
flannMat = cv2.FlannBasedMatcher_create()

while True:
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    frame = frame1.copy()

    # ORB模型检测关键点
    kpts1, des1 = orb.detectAndCompute(gray1, mask=None)
    kpts2, des2 = orb.detectAndCompute(gray2, mask=None)
    cv2.drawKeypoints(frame1, kpts1, frame1, (0, 0, 255),
                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.drawKeypoints(frame2, kpts2, frame2, (0, 0, 255),
                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 暴力匹配特征
    # matches = bfMat.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # frame = cv2.drawMatches(frame1, kpts1, frame2, kpts2, matches[:10],
    #                          frame, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # FLANN匹配
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    kMatches = flannMat.knnMatch(des1, des2, k=2)
    matMask = [[0, 0] for i in range(len(kMatches))]
    # David G. Lowe's比例测试
    for i, (m1, m2) in enumerate(kMatches):
        if m1.distance < 0.7 * m2.distance:
            matMask[i] = [1, 0]
    frame = cv2.drawMatchesKnn(frame1, kpts1, frame2, kpts2, kMatches, None,
                               matchesMask=matMask, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    cv2.imshow('Display', frame)

    keyCode = cv2.waitKey(1)
    if keyCode == 27:
        break


cap1.release()
cap2.release()
cv2.destroyAllWindows()


