import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r'.\resources\cartoon1.jpg', cv2.IMREAD_COLOR)
imgCp1 = img.copy()
imgCp2 = img.copy()
imgCp3 = img.copy()
imgCp4 = img.copy()

# 图片旋转30度
h, w = img.shape[:2]
shrink = cv2.resize(img, (w//2, h//2))
M = cv2.getRotationMatrix2D((w/2, h/2), 30, 1)
M += np.float32([[0, 0, (h-w)/2 + 150], [0, 0, (w-h)/2 - 50]])
imgRot = cv2.warpAffine(img, M, (w+50, h+100))

gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(imgRot, cv2.COLOR_BGR2GRAY)

# ORB模型计算两张图片的关键点
orb = cv2.ORB_create()
kpts1, des1 = orb.detectAndCompute(gray1, mask=None)
kpts2, des2 = orb.detectAndCompute(gray2, mask=None)
cv2.drawKeypoints(img, kpts1, img, color=(0, 0, 255),
                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.drawKeypoints(imgRot, kpts2, imgRot, color=(0, 0, 255),
                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
print('image shape:', gray1.shape)
print('Descriptor shape:', des1.shape)
print('keypoints Number:', len(kpts1))  # 关键点个数等于描述符行数
print('Keypoint[0] angle:', kpts1[0].angle)
print('Keypoint[0] class_id:', kpts1[0].class_id)
print('Keypoint[0] octave:', kpts1[0].octave)
print('Keypoint[0] pt:', kpts1[0].pt)
print('Keypoint[0] response:', kpts1[0].response)
print('Keypoint[0] size:', kpts1[0].size)

# descriptors.convertTo(descriptors,CV_32F);
# 使用暴力匹配方法匹配两张图片的描述符
bfMat = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
matches1 = bfMat.match(des1, des2)
print('匹配个数：', len(matches1))
print('match[0] distance：', matches1[0].distance)
print('match[0] imgIdx：', matches1[0].imgIdx)
print('match[0] queryIdx：', matches1[0].queryIdx)
print('match[0] trainIdx：', matches1[0].trainIdx)
# 把匹配到的关键点在图中的坐标找到
srcPts = np.float32([kpts1[m.queryIdx].pt for m in matches1])
print('points shape:', srcPts.shape)
srcPts = srcPts.reshape(-1, 1, 2)
print('after reshape, points shape:', srcPts.shape)

dists = [d.distance for d in matches1]
plt.hist(dists)
plt.show()
matches1 = sorted(matches1, key=lambda x: x.distance)
imgCp1 = cv2.drawMatches(img, kpts1, imgRot, kpts2, matches1[:10],
                        imgCp1, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# 使用knnMatch，每个特征查看3个最佳匹配
# bfMat = cv2.BFMatcher_create(cv2.NORM_HAMMING)
kMatches = bfMat.knnMatch(des1, des2, k=3)
print(len(kMatches))
print(len(kMatches[0]))
imgCp2 = cv2.drawMatchesKnn(img, kpts1, imgRot, kpts2, kMatches[:10],
                            imgCp2, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 仅取出3个匹配中的第2个
# firstMatch = []
# for m in kMatches:
#     firstMatch.append(m[1])
# firstMatch = sorted(firstMatch, key=lambda x: x.distance)
# imgCp2 = cv2.drawMatches(img, kpts1, imgRot, kpts2, firstMatch[:10],
#                         imgCp2, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


# 使用FLANN匹配方法匹配两张图片
indPars = dict(algorithm=0, trees=5)
# searchPars = dict(checks=50)
searchPars = {}
flannMat = cv2.FlannBasedMatcher_create()
# flannMat = cv2.FlannBasedMatcher(indPars, searchPars)
des1 = np.float32(des1)
des2 = np.float32(des2)
matches2 = flannMat.match(des1, des2)
matches2 = sorted(matches2, key=lambda x: x.distance)
imgCp3 = cv2.drawMatches(img, kpts1, imgRot, kpts2, matches2[:10],
                        imgCp3, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 每个特征取2个最佳匹配
kMatches = flannMat.knnMatch(des1, des2, k=2)
matMask = [[0, 0] for i in range(len(kMatches))]
# David G. Lowe's比例测试，将两个最佳匹配距离差值足够大的挑出来
for i, (m1, m2) in enumerate(kMatches):
    if m1.distance < 0.7 * m2.distance:
        matMask[i] = [1, 0]
imgCp4 = cv2.drawMatchesKnn(img, kpts1, imgRot, kpts2, kMatches, imgCp4,
                            matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                            matchesMask=matMask, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.namedWindow('matches2', cv2.WINDOW_NORMAL)
cv2.imshow('matches2', cv2.vconcat((imgCp3, imgCp4)))

cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
# cv2.imshow('matches', imgCp1)
cv2.imshow('matches', cv2.vconcat((imgCp1, imgCp2)))

cv2.waitKey(0)
cv2.destroyAllWindows()


