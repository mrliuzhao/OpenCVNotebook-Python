import cv2
import numpy as np


img = cv2.imread(r".\resources\card.jpg", cv2.IMREAD_COLOR)
rows, cols, channels = img.shape

# 卡片的四个角点，非共线的四个点才能唯一确定透视变换
pts1 = np.float32([[148, 80], [437, 114], [94, 247], [423, 288]])
# 变换后分别在左上、右上、左下、右下四个点
pts2 = np.float32([[0, 0], [320, 0], [0, 178], [320, 178]])

# 透视变换
M = cv2.getPerspectiveTransform(pts1, pts2)
imgPersp = cv2.warpPerspective(img, M, (320, 178))

gray1 = cv2.cvtColor(imgPersp, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB模型检测关键点
orb = cv2.ORB_create()
kpts1, des1 = orb.detectAndCompute(gray1, mask=None)
kpts2, des2 = orb.detectAndCompute(gray2, mask=None)
cv2.drawKeypoints(imgPersp, kpts1, imgPersp, color=(0, 0, 255),
                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.drawKeypoints(img, kpts2, img, color=(0, 0, 255),
                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# FLANN匹配关键点
indPars = dict(algorithm=0, trees=5)
searchPars = dict(checks=50)
flannMat = cv2.FlannBasedMatcher(indPars, searchPars)
des1 = np.float32(des1)
des2 = np.float32(des2)
kMatches = flannMat.knnMatch(des1, des2, k=2)
# David G. Lowe's比例测试
goodMat = []
for m1, m2 in kMatches:
    if m1.distance < 0.7 * m2.distance:
        goodMat.append(m1)

print('Good Match Number:', len(goodMat))

src_pts = np.float32([kpts1[m.queryIdx].pt for m in goodMat]).reshape(-1, 1, 2)
dst_pts = np.float32([kpts2[m.trainIdx].pt for m in goodMat]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
print('M type:', type(M))
print('M shape:', M.shape)
print('mask type:', type(mask))
print('mask shape:', mask.shape)

matMask = mask.ravel().tolist()
print('type of matMask:', type(matMask))

# 将四个角点转换到原图，并画出
h, w = imgPersp.shape[:2]
cornerPts = np.float32([[0, 0],
                        [0, h-1],
                        [w, h],
                        [w-1, 0]]).reshape(-1, 1, 2)
cornerPts = cv2.perspectiveTransform(cornerPts, M)
cornerPts = cornerPts.astype(np.int32)
img = cv2.polylines(img, [cornerPts], True, (0, 0, 255), 2, cv2.LINE_AA)
print(cornerPts)

imgMat = cv2.drawMatches(imgPersp, kpts1, img, kpts2, goodMat, None,
                         matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                         matchesMask=matMask, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
cv2.imshow('matches', imgMat)


cv2.waitKey(0)
cv2.destroyAllWindows()

