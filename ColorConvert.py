import cv2
import numpy as np

# 读取原始图像并展示
imgOrigin = cv2.imread(r".\resources\Luffy1.jpeg", cv2.IMREAD_UNCHANGED)
cv2.namedWindow('Origin Picture', cv2.WINDOW_NORMAL)
cv2.imshow("Origin Picture", imgOrigin)

# 转换为灰度图像
# 灰度化公式：gray=R*0.299+G*0.587+B*0.114
imgGray = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Picture Gray', cv2.WINDOW_NORMAL)
cv2.imshow("Picture Gray", imgGray)

# 从BGR -> RGB
imgInverse = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2RGB)
cv2.namedWindow('Picture Inverse', cv2.WINDOW_NORMAL)
cv2.imshow("Picture Inverse", imgInverse)

# 转换为HSV模式，注意imshow仍认为图片在BGR空间
imgHSV = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2HSV)
cv2.imshow("Picture in HSV", imgHSV)

# 根据颜色范围追踪筛选
# OpenCV中色调H范围为[0,179]（0~360°除以2），饱和度S是[0,255]，明度V是[0,255]
# 红色0°——H=0；绿色120°——H=60；蓝色240°——H=120
# 黄色的范围，不同光照条件下不一样，可灵活调整
lower_yellow = np.array([10, 60, 60])
upper_yellow = np.array([50, 255, 255])
# 介于lower/upper之间为白色（255），其余为黑色（0）
mask = cv2.inRange(imgHSV, lower_yellow, upper_yellow)
cv2.imshow("Picture mask", mask)
# 按位与提取原图中黄色部分
redRegion = cv2.bitwise_and(imgOrigin, imgOrigin, mask=mask)
cv2.imshow("Picture Yellow Region", redRegion)

cv2.waitKey(0)
cv2.destroyAllWindows()
