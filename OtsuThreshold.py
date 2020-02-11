import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread(r".\resources\noisy.jpg", cv2.IMREAD_GRAYSCALE)

# 固定阈值法，手动设置阈值为100，常称之为经验阈值，需要多次尝试获得
ret1, thBin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# Otsu法（大津法——最大类间方差法）计算出全局最佳阈值，因此第二个参数忽略。
# Otsu法最适用于双波峰图像，即图像直方图有两个波峰，计算出的阈值即为两个波峰之间的谷底
start = cv2.getTickCount()
ret2, thOtsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
end = cv2.getTickCount()
print('Official Ostu cost:', (end - start) / cv2.getTickFrequency())
# Ostu中类间方差：g = p0(u0-u)^2 + p1(u1-u)^2 = p0p1(u0-u1)^2 （类似方差分析中的效应平方和、组间方差）
# p0为背景像素比例，p1为前景像素比例；u0为背景像素灰度值的平均值，u1为前景像素灰度值的平均值，u为总平均值
# 遍历寻找使类间方差最大化的阈值

# 尝试自己实现OTSU法寻找最佳全局阈值，效率极低
imgArr = img.ravel()
totalCount = len(imgArr)
totalSum = sum(imgArr)
maxDelta = None
ostuThreshold = None
start = cv2.getTickCount()
for t in range(256):
    back = list(filter(lambda x: x < t, imgArr))
    # fore = list(filter(lambda x: x >= t, imgArr))
    u0 = 0 if len(back) == 0 else np.mean(back)
    n1 = totalCount - len(back)
    u1 = 0 if n1 == 0 else (totalSum - sum(back)) / n1
    p0 = len(back) / totalCount
    p1 = n1 / totalCount
    delta = p0*p1*((u0-u1)**2)
    if maxDelta is None or maxDelta < delta:
        maxDelta = delta
        ostuThreshold = t
ret1, thSelf = cv2.threshold(img, ostuThreshold, 255, cv2.THRESH_BINARY)
end = cv2.getTickCount()
print('My Ostu cost:', (end - start) / cv2.getTickFrequency())

# 先进行高斯滤波，再使用Otsu阈值法
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, thOtsuGaus = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 三角形二值法计算最佳阈值，最适用于单个波峰，最开始用于医学分割细胞等
ret4, thTri = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

images = [img, 0, thBin,
          img, 0, thOtsu,
          img, 0, thSelf,
          blur, 0, thOtsuGaus,
          img, 0, thTri]
titles = ['Original', 'Histogram', 'Global(v=100)',
          'Original', 'Histogram', "Otsu's Threshold",
          'Original', 'Histogram', "Otsu by myself",
          'Gaussian filtered Image', 'Histogram', "Otsu's Threshold",
          'Original', 'Histogram', "Triangle's"]

for i in range(4):
    # 绘制原图
    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3], fontsize=8)
    plt.xticks([]), plt.yticks([])

    # 绘制直方图plt.hist，ravel函数将数组降成一维
    plt.subplot(4, 3, i * 3 + 2)
    plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1], fontsize=8)
    plt.xticks([]), plt.yticks([])

    # 绘制阈值图
    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()



