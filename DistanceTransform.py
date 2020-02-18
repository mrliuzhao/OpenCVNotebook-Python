import cv2
import numpy as np


# img = cv2.imread(r".\resources\j.bmp", cv2.IMREAD_GRAYSCALE)
img = np.zeros((500, 500), np.uint8)
# 在黑色背景画两个方框
img[100:150, 100:150] = 255
img[300:350, 400:450] = 255

# L2距离，即欧氏距离，两点间直线距离。sqrt{(x1-x2)^2+(y1-y2)^2)}
distEuc = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=3)
# L1距离，即曼哈顿距离，国际象棋中车的走法，一次只能走一格，所需步数。|x1-x2|+|y1-y2|
distD4 = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=3)
# 切比雪夫距离，即按国际象棋中王的走法，两点间所需步数。max{|x1-x2|,|y1-y2|}
distD8 = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=3)
print(np.max(distEuc))
print(np.max(distD4))
# 需要归一化距离矩阵
cv2.normalize(distEuc, distEuc, 0, 1, cv2.NORM_MINMAX)
cv2.normalize(distD4, distD4, 0, 1, cv2.NORM_MINMAX)
cv2.normalize(distD8, distD8, 0, 1, cv2.NORM_MINMAX)
print(np.max(distEuc))
print(np.max(distD4))
cv2.imshow('Distance Transform', np.vstack((np.hstack((img, distEuc)), np.hstack((distD4, distD8)))))


cv2.waitKey(0)
cv2.destroyAllWindows()

