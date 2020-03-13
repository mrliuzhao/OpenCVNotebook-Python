import cv2
import numpy as np
import matplotlib.pyplot as plt


cap1 = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(1)

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
orb = cv2.ORB_create()
bfMat = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
flannMat = cv2.FlannBasedMatcher_create()

# vrimg = cv2.imread(r".\resources\faces\lz\0.pgm", cv2.IMREAD_COLOR)
vrimg = cv2.imread(r".\resources\VR2.png", cv2.IMREAD_COLOR)
vrgray = cv2.cvtColor(vrimg, cv2.COLOR_BGR2GRAY)
vrkpts, vrdes = orb.detectAndCompute(vrgray, mask=None)
cv2.drawKeypoints(vrimg, vrkpts, vrimg, (0, 0, 255),
                  cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
h, w = vrimg.shape[:2]
vrCorner = np.float32([[0, 0],
                       [0, h - 1],
                       [w, h],
                       [w - 1, 0]]).reshape(-1, 1, 2)
# print('VR Corner:', vrCorner)
cornerPts = None
while True:
    _, frame1 = cap1.read()
    # _, frame2 = cap2.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # ORB模型检测关键点
    kpts1, des1 = orb.detectAndCompute(gray1, mask=None)
    cv2.drawKeypoints(frame1, kpts1, frame1, (0, 0, 255),
                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # kpts2, des2 = orb.detectAndCompute(gray2, mask=None)
    # cv2.drawKeypoints(frame2, kpts2, frame2, (0, 0, 255),
    #                   cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 暴力匹配特征
    # matches = bfMat.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # frame = cv2.drawMatches(frame1, kpts1, frame2, kpts2, matches[:10],
    #                          frame, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # FLANN匹配
    des1 = np.float32(des1)
    # des2 = np.float32(des2)
    des2 = np.float32(vrdes)
    kMatches = flannMat.knnMatch(des1, des2, k=2)
    # David G. Lowe's比例测试
    goodMat = []
    for m1, m2 in kMatches:
        if m1.distance < 0.7 * m2.distance:
            goodMat.append(m1)
    # frame = cv2.drawMatchesKnn(frame1, kpts1, frame2, kpts2, kMatches, None,
    #                            matchesMask=matMask, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # 将VR区域绘制出来
    if len(goodMat) > 4:
        src_pts = np.float32([vrkpts[m.trainIdx].pt for m in goodMat]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts1[m.queryIdx].pt for m in goodMat]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is not None:
            cornerPts = cv2.perspectiveTransform(vrCorner, M)
            cornerPts = cornerPts.astype(np.int32)
            # print(cornerPts)

    if cornerPts is not None:
        frame1 = cv2.polylines(frame1, [cornerPts], True, (0, 0, 255), 2, cv2.LINE_AA)
    # 绘制匹配点
    frame = cv2.drawMatches(frame1, kpts1, vrimg, vrkpts, goodMat, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    cv2.imshow('Display', frame)

    keyCode = cv2.waitKey(1)
    if keyCode == 27:
        break


cap1.release()
# cap2.release()
cv2.destroyAllWindows()


