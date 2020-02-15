import cv2
import numpy as np

img = cv2.imread(r".\resources\sudoku.jpg", cv2.IMREAD_GRAYSCALE)

# Canny算法进行边缘检测，Canny边缘检测算法的大致步骤：
# 1. 使用5*5的高斯卷积核降噪；
# 2. 使用3*3的Sobel梯度滤波器计算0, 45, 90, 135度四个方向上的梯度
# 3. 非极大值抑制（Non-Maximum Suppression，NMS）排除非边缘的像素，仅保留一些细线条作为候选边缘
# 4. 滞后阈值，方法中两个threshold参数即分别为该步骤中的高低阈值，推荐的高低阈值比在2:1 ~ 3:1之间。像素值超过高阈值时被保留；小于低阈值时被排除；在高低阈值之间时，仅在该像素连接到一个高于高阈值的像素时保留。
edges = cv2.Canny(img, threshold1=30, threshold2=70)
edges2 = cv2.Canny(img, threshold1=30, threshold2=70, L2gradient=True)
cv2.imshow('canny', np.hstack((img, edges, edges2)))

# 以canny检测出的边缘作为mask显示原图像
edgeImg = cv2.bitwise_and(img, img, mask=edges)
edgesInv = cv2.bitwise_not(edges)
edgeImgInv = cv2.bitwise_and(img, img, mask=edgesInv)
cv2.imshow('edge in Origin image', np.hstack((img, edgeImg, edgeImgInv)))


def Nothing(x):
    pass


cv2.namedWindow('Canny Edge Detection', cv2.WINDOW_NORMAL)
cv2.createTrackbar('maxThreshold', 'Canny Edge Detection', 0, 255, Nothing)
cv2.createTrackbar('minThreshold', 'Canny Edge Detection', 0, 255, Nothing)
cv2.namedWindow('Canny After Binarization', cv2.WINDOW_NORMAL)
cv2.createTrackbar('maxThreshold', 'Canny After Binarization', 0, 255, Nothing)
cv2.createTrackbar('minThreshold', 'Canny After Binarization', 0, 255, Nothing)
# 尝试先阈值法二分图像，再检测边缘
imgTh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)

while True:
    if cv2.waitKey(1) == 27:
        break

    maxTh = cv2.getTrackbarPos('maxThreshold', 'Canny Edge Detection')
    minTh = cv2.getTrackbarPos('minThreshold', 'Canny Edge Detection')
    edges = cv2.Canny(img, threshold1=maxTh, threshold2=minTh)
    edges2 = cv2.Canny(img, threshold1=maxTh, threshold2=minTh, L2gradient=True)
    cv2.imshow('Canny Edge Detection', np.hstack((img, edges, edges2)))

    # 由于原图二值化，滞后阈值不起任何作用，并且一些未被消除的胡椒盐噪声一定会被检测成为边缘
    maxTh2 = cv2.getTrackbarPos('maxThreshold', 'Canny After Binarization')
    minTh2 = cv2.getTrackbarPos('minThreshold', 'Canny After Binarization')
    edgesTh = cv2.Canny(imgTh, threshold1=minTh2, threshold2=maxTh2)
    edgesTh2 = cv2.Canny(imgTh, threshold1=minTh2, threshold2=maxTh2, L2gradient=True)
    cv2.imshow('Canny After Binarization', np.hstack((imgTh, edgesTh, edgesTh2)))


cv2.destroyAllWindows()
