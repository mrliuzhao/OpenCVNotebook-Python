import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r".\resources\cat2.jpg", cv2.IMREAD_COLOR)
cv2.imshow('origin', img)

# 固定值padding边框，上下左右均添加一个border，统一都填充0也称为zero padding
imgPad = cv2.copyMakeBorder(img, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_CONSTANT, value=0)
cv2.imshow('Padding zeros', imgPad)

# 平均卷积核
kernel = np.ones((7, 7), np.float32) / 49
print(kernel)
# 卷积操作，-1表示depth与原图相同
unifFilter = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_DEFAULT)

# 锐化卷积核
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]], np.float32)
print(kernel)
sharpen = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_DEFAULT)

# OpenCV自带的blur函数就是均值滤波
blur = cv2.blur(img, (9, 9), borderType=cv2.BORDER_CONSTANT)

# 方框滤波器
boxFilter = cv2.boxFilter(img, -1, (5, 5), normalize=True, borderType=cv2.BORDER_REFLECT)

# 高斯滤波器
gaussian = cv2.GaussianBlur(img, (7, 7), 3)
print(cv2.getGaussianKernel(7, 3))

# 中值滤波器，适用于去除斑点状的噪声，如椒盐噪声。非线性滤波
median = cv2.medianBlur(img, 3)
cv2.imshow('median', median)

# 双边滤波，该滤波器可以在消除噪声、模糊图片的同时，尽可能地保留边缘信息（即尽可能让物体边缘仍锐利）。非线性滤波
# d: 第二个参数表示滤波器大小，越大越慢，官方建议实时运算在5以内，线下应用设置d=9即可。为负数时则会自动根据sigmaSpace计算
# sigmaColor：色彩方差
# sigmaSpace：色彩在空间上的方差
# 上述两个sigma值，可以简单设置一样，值越大越会使照片显得“卡通化”
bilateral = cv2.bilateralFilter(img, -1, 120, 120)
cv2.imshow('bilateral', bilateral)

titles = ['Original', 'Uniform Filter\n(kernel size = 7)', 'Sharpen',
          'Uniform Blur\n(kernel size = 9)', 'Box Filter\n(kernel size = 5)',
          'Gaussian Filter\n(kernel size = 5, sigma = 3)', 'Median Filter\n(kernel size = 5)',
          'Bilateral Filter']
images = [img, unifFilter, sharpen, blur, boxFilter, gaussian, median, bilateral]

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
