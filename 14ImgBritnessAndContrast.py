import cv2
import numpy as np

img = cv2.imread(r".\resources\Luffy1.jpeg", cv2.IMREAD_COLOR)

# OpenCV中，g(x)=αf(x) + β
# 其中α(>0)称为增益（gain）、β称为偏置值（Bias）
# 对比度调整：图像暗处像素强度变低，图像亮处像素强度变高，从而拉大中间某个区域范围的显示精度，需要通过α和β配合一起控制
# 亮度调整：图像像素强度整体变高/变低。仅需通过β控制

# 降低对比度例子，a = 0.6
# 调整中心：灰度125(即映射曲线经过(125,125)这个点)
# 可以计算出b = 125*(1-0.6)。 注意这里的b与亮度没有任何关系，仅仅用于对比度调整
a = 0.6
b = 125*(1-0.6)
# 注意Numpy中使用clip函数控制最后取值范围，小于0的为0，大于255的为255
res = np.uint8(np.clip((a * img + b), 0, 255))
tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
cv2.imshow('image contrast down', tmp)

# 增加对比度例子，a = 1.68
# 调整中心：灰度125(即映射曲线经过(125,125)这个点)
# 可以计算出b = 125*(1-1.68)。 注意这里的b与亮度没有任何关系，仅仅用于对比度调整
a = 1.68
b = 125*(1-1.68)
res = np.uint8(np.clip((a * img + b), 0, 255))
tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
cv2.imshow('image contrast up', tmp)


def nothing(x):
    pass


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('a', 'image', 0, 300, nothing)
cv2.createTrackbar('b', 'image', 0, 100, nothing)

while True:
    if cv2.waitKey(1) == 27:
        break

    a = cv2.getTrackbarPos('a', 'image')
    a = a / 100
    b = cv2.getTrackbarPos('b', 'image')

    res = np.uint8(np.clip((a * img + b), 0, 255))  # clip函数控制取值范围
    tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
    cv2.imshow('image', tmp)


cv2.destroyAllWindows()
