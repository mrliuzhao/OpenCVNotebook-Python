import cv2
import numpy as np


# 回调函数，x表示滑块的位置，本例暂不使用
def nothing(x):
    pass


imgHSV = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')

# 创建HSV三个滑动条
# OpenCV中色调H范围为[0,179]（0~360°除以2），饱和度S是[0,255]，明度V是[0,255]
# 红色0°——H=0；绿色120°——H=60；蓝色240°——H=120
cv2.createTrackbar('H', 'image', 0, 179, nothing)
cv2.createTrackbar('S', 'image', 0, 255, nothing)
cv2.createTrackbar('V', 'image', 0, 255, nothing)

while True:
    if cv2.waitKey(1) == 27:
        break

    # 获取滑块的值
    h = cv2.getTrackbarPos('H', 'image')
    s = cv2.getTrackbarPos('S', 'image')
    v = cv2.getTrackbarPos('V', 'image')
    # 设定img的颜色
    imgHSV[:] = [h, s, v]
    imgBGR = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
    cv2.imshow('image', imgBGR)

cv2.destroyAllWindows()

