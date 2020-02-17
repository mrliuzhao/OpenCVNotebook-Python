import cv2
import numpy as np


img = cv2.imread(r".\resources\sudoku.jpg", cv2.IMREAD_GRAYSCALE)
imgBi = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
imgBi = cv2.bitwise_not(imgBi)
cv2.imshow('Origin', imgBi)

# 形态学操作一般作用于二值化图，来连接相邻的元素或分离成独立的元素
# 每次操作仅关注核大小的区域，且仅计算核中不为0的部分。全为1的核则表示关注核大小区域中的所有元素。
kernel = np.ones((3, 3), np.uint8)
# 通过getStructuringElement方法获取不同形状的形态学操作结构（即核）
kernelRect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构，等同于全1的矩阵
kernelEllipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
kernelCross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字形结构
print(kernelRect)
print(kernelCross)
print(kernelEllipse)
# 腐蚀的效果是把图片"变瘦"，其原理是在原图的小区域内取局部最小值
erosion = cv2.erode(imgBi, kernel)
erosionRect = cv2.erode(imgBi, kernelRect)
erosionEllipse = cv2.erode(imgBi, kernelEllipse)
erosionCross = cv2.erode(imgBi, kernelCross)
r1 = np.hstack((erosion, erosionRect))
r2 = np.hstack((erosionEllipse, erosionCross))
cv2.imshow('Erosion', np.vstack((r1, r2)))

# 膨胀与腐蚀相反，取的是局部最大值，效果是把图片”变胖”
kernelRect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernelEllipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernelCross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
dilate = cv2.dilate(imgBi, kernel)
dilateRect = cv2.dilate(imgBi, kernelRect)
dilateEllipse = cv2.dilate(imgBi, kernelEllipse)
dilateCross = cv2.dilate(imgBi, kernelCross)
r1 = np.hstack((dilate, dilateRect))
r2 = np.hstack((dilateEllipse, dilateCross))
cv2.imshow('Dilation', np.vstack((r1, r2)))

# 先腐蚀后膨胀叫开运算，主要用于分离物体、消除物体外的小区域，比如物体外的椒盐噪点
img = cv2.imread(r'./resources/j_noise_out.bmp', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', np.hstack((img, opening)))

# 反之，先膨胀后腐蚀称为闭运算，主要用于消除/闭合物体中的小黑洞，比如物体内的椒盐噪点
img = cv2.imread(r'./resources/j_noise_in.bmp', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', np.hstack((img, closing)))

# 其他形态学操作
# 梯度运算，结果为膨胀图 - 腐蚀图，获得物体轮廓
img = cv2.imread(r'./resources/j.bmp', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilate = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
erode = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('Gradient', np.vstack((np.hstack((img, gradient)), np.hstack((dilate, erode))))
)

# TopHat运算，结果为原图 - 开运算图，相当于提取了物体外部的噪点
img = cv2.imread(r'./resources/j_noise_out.bmp', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('Top Hat', np.hstack((img, opening, tophat)))

# BlackHat运算，结果为闭运算图 - 原图，相当于提取了物体内部的噪点
img = cv2.imread(r'./resources/j_noise_in.bmp', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('Black Hat', np.hstack((img, closing, blackhat)))

# Hit and Miss运算，不仅考虑前景，也考虑背景，常用于匹配指定模式
# 即核中-1表示需要跟背景匹配，1表示跟前景（物体）匹配，0才表示不关心
img = np.zeros((500, 500), np.uint8)
# 在黑色背景画两个方框
img[100:150, 100:150] = 255
img[300:350, 400:450] = 255
# 把左上角的元素选择出来
kernel = np.array([[-1, -1, -1],
                   [-1, 1, 0],
                   [-1, 0, 0]], np.float32)
hitMiss1 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
# 再把右下角的元素选择出来
kernel = np.array([[0, 0, -1],
                   [0, 1, -1],
                   [-1, -1, -1]], np.float32)
hitMiss2 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
hitMiss = cv2.bitwise_or(hitMiss1, hitMiss2)  # 合并左上角和右下角元素
cv2.imshow('Hit And Miss', np.hstack((img, hitMiss)))


cv2.waitKey(0)
cv2.destroyAllWindows()


