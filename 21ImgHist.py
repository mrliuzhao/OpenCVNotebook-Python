import cv2
import numpy as np
from matplotlib import pyplot as plt

bins = np.arange(256).reshape(256, 1)


# OpenCV官方提供的直方图绘制示例
def hist_curve(im):
    h = np.zeros((300, 256, 3))
    if len(im.shape) == 2:
        color = [(255, 255, 255)]
    elif im.shape[2] == 3:
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([im], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins, hist)))
        cv2.polylines(h, [pts], False, col)
    y = np.flipud(h)
    return y


def hist_lines(im):
    h = np.zeros((300, 256, 3))
    if len(im.shape) != 2:
        print("hist_lines applicable only for grayscale images")
        # print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im], [0], None, [256], [0, 256])
    cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    for x, y in enumerate(hist):
        cv2.line(h, (x, 0), (x, y), (255, 255, 255))
    y = np.flipud(h)
    return y


img = cv2.imread(r".\resources\dog2.jpg", cv2.IMREAD_GRAYSCALE)

start = cv2.getTickCount()
# OpenCV计算图片直方图的函数
# channels即为统计哪个通道的直方图；histSize即为直方图bins个数；ranges为统计哪个范围内的值
histCV = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
end = cv2.getTickCount()
print('OpenCV计算直方图用时：{} s'.format((end - start) / cv2.getTickFrequency()))

start = cv2.getTickCount()
# Numpy也有计算图片直方图的方法
histNp, bins = np.histogram(img.ravel(), bins=256, range=[0, 256])
end = cv2.getTickCount()
print('Numpy计算直方图用时：{} s'.format((end - start) / cv2.getTickFrequency()))

plt.subplot(2, 1, 1)
plt.plot(histCV)
plt.title('Histogram by OpenCV', fontsize=8)
plt.subplot(2, 1, 2)
plt.plot(histNp)
plt.title('Histogram by Numpy', fontsize=8)
plt.show()

# 直方图均衡化（Histogram Equalization，HE）
# 将直方图尽可能拉平为均匀分布，效果为增强对比度，但同时也会增强噪声
equ = cv2.equalizeHist(img)
histEqu = cv2.calcHist([equ], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# 对比度受限的自适应直方图均衡化(Contrast Limited Adaptive Histogram Equalization，CLAHE)
# 该方法在每个小区域内对图像进行直方图均衡化，并且为了防止噪点被放大，提供对比度限制参数，超过限制的不进行均衡化
# clipLimit参数即为对比度限制阈值；tileGridSize即为进行HE的区块大小
clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
cla = clahe.apply(img)
histCla = cv2.calcHist([cla], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# 对比展示两种均衡化方法
cv2.imshow('Compare Image after equalization', cv2.hconcat((img, equ, cla)))
plt.subplot(3, 1, 1)
plt.plot(histCV)
plt.title('Hist of Origin', fontsize=8)
plt.subplot(3, 1, 2)
plt.plot(histEqu)
plt.title('Hist After Equalization', fontsize=8)
plt.subplot(3, 1, 3)
plt.plot(histCla)
plt.title('Hist After CLAHE', fontsize=8)
plt.show()


# 尝试对彩色图片进行直方图均衡
# img = cv2.imread(r".\resources\Luffy1.jpeg", cv2.IMREAD_COLOR)
img = cv2.imread(r".\resources\dog1.jpg", cv2.IMREAD_COLOR)

# 通过图像增益（Gain）和偏置（Bias）增强对比度
res = np.uint8(np.clip((1.68 * img - 85.0), 0, 255))

# 拆分BGR三通道，分别对三通道进行直方图均衡
b, g, r = cv2.split(img)
equB = cv2.equalizeHist(b)
equG = cv2.equalizeHist(g)
equR = cv2.equalizeHist(r)
equCol = cv2.merge((equB, equG, equR))

clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
claB = clahe.apply(b)
claG = clahe.apply(g)
claR = clahe.apply(r)
claCol = cv2.merge((claB, claG, claR))

cv2.namedWindow('Colorful Image after equalization', cv2.WINDOW_NORMAL)
cv2.imshow('Colorful Image after equalization', cv2.hconcat((img, res, equCol, claCol)))

# 可以看出对颜色分量进行直方图均衡没有任何对比度增强效果，故还是应该对图像中的灰度/亮度部分进行直方图均衡化
# 因此考虑在YCbCr色彩空间进行直方图均衡
# Y是指亮度(luma)分量(灰阶值)，Cb指蓝色色度分量，而Cr指红色色度分量
# 综上，仅对第一个通道——Y进行直方图均衡即可
# 转换至YCbCr色彩空间，并拆分Y、Cb、Cr三通道，仅对Y通道进行直方图均衡
imgYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(imgYCrCb)
equY = cv2.equalizeHist(y)
equCol = cv2.merge((equY, cr, cb))
equCol = cv2.cvtColor(equCol, cv2.COLOR_YCrCb2BGR)

clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
claY = clahe.apply(y)
claCol = cv2.merge((claY, cr, cb))
claCol = cv2.cvtColor(claCol, cv2.COLOR_YCrCb2BGR)

cv2.namedWindow('Equalize In YCrCb', cv2.WINDOW_NORMAL)
cv2.imshow('Equalize In YCrCb', cv2.hconcat((img, res, equCol, claCol)))


# 再尝试在HSV色彩空间中对V通道进行直方图均衡
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(imgHSV)
equV = cv2.equalizeHist(v)
equCol = cv2.merge((h, s, equV))
equCol = cv2.cvtColor(equCol, cv2.COLOR_HSV2BGR)

clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(8, 8))
claY = clahe.apply(v)
claCol = cv2.merge((h, s, claY))
claCol = cv2.cvtColor(claCol, cv2.COLOR_HSV2BGR)

cv2.namedWindow('Equalize In HSV', cv2.WINDOW_NORMAL)
cv2.imshow('Equalize In HSV', cv2.hconcat((img, res, equCol, claCol)))


cv2.waitKey(0)
cv2.destroyAllWindows()

