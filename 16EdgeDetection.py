import cv2
import numpy as np

img = cv2.imread(r".\resources\sudoku.jpg", cv2.IMREAD_COLOR)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel垂直边缘提取滤波器，查看图像在x方向上梯度，相当于在x方向上求导
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
sobelV = cv2.filter2D(imgGray, -1, kernel, borderType=cv2.BORDER_DEFAULT)

# Sobel水平边缘提取滤波器，查看图像在y方向上梯度，相当于在y方向上求导
kernel = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], np.float32)
sobelH = cv2.filter2D(imgGray, -1, kernel, borderType=cv2.BORDER_DEFAULT)
cv2.imshow('sobel by filter', np.hstack((imgGray, sobelV, sobelH)))

# 使用OpenCV中的Sobel算子，dx和dy即为x和y方向上导数的阶
sobelx = cv2.Sobel(imgGray, ddepth=-1, dx=1, dy=0, ksize=3)  # 只计算x方向的1阶导数，即提取垂直方向边缘
sobely = cv2.Sobel(imgGray, ddepth=-1, dx=0, dy=1, ksize=3)  # 只计算y方向的1阶导数，即提取水平方向边缘
cv2.imshow('sobel by OpenCV', np.hstack((imgGray, sobelx, sobely)))

# Prewitt算子
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], np.float32)
prewittV = cv2.filter2D(imgGray, -1, kernel, borderType=cv2.BORDER_DEFAULT)
# Scharr算子
kernel = np.array([[-3, 0, 3],
                   [-10, 0, 10],
                   [-3, 0, 3]], np.float32)
scharrV = cv2.filter2D(imgGray, -1, kernel, borderType=cv2.BORDER_DEFAULT)
# OpenCV也有自带Scharr函数
scharrV	= cv2.Scharr(imgGray, ddepth=-1, dx=1, dy=0, borderType=cv2.BORDER_DEFAULT)
# scharrV = cv2.Sobel(imgGray, ddepth=-1, dx=1, dy=0, ksize=cv2.FILTER_SCHARR)  # ksize为FILTER_SCHARR即为Scharr算子

# 三种一阶边缘检测对比
cv2.imshow('Compare Prewitt Sobel and Scharr', np.hstack((imgGray, prewittV, sobelV, scharrV)))

# Laplacian算子，是对图像进行二阶求导来检测边缘，即二阶边缘检测
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], np.float32)
laplacian1 = cv2.filter2D(imgGray, -1, kernel, borderType=cv2.BORDER_DEFAULT)
# 扩展算子，增加斜对角方向的边缘检测
kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]], np.float32)
laplacian2 = cv2.filter2D(imgGray, -1, kernel, borderType=cv2.BORDER_DEFAULT)
# 使用OpenCV自带Laplacian函数
laplacian3 = cv2.Laplacian(imgGray, ddepth=-1, ksize=3, borderType=cv2.BORDER_DEFAULT)
cv2.imshow('Laplacian', np.hstack((imgGray, laplacian1, laplacian2, laplacian3)))


cv2.waitKey(0)
cv2.destroyAllWindows()
