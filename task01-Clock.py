import cv2
import datetime
import numpy as np
import math

img = np.zeros((1024, 1024, 3), np.uint8)
img[:] = (255, 255, 255)
height, width, channels = img.shape
center_x = int(width/2)
center_y = int(height/2)
cv2.namedWindow('Clock', cv2.WINDOW_NORMAL)

rFar = int(height/2-3)
# 绘制表盘
cv2.circle(img, center=(center_x, center_y), radius=rFar,
           thickness=5, lineType=cv2.LINE_AA, color=(0, 0, 0))

rNearBold = rFar - 40
rNear = rFar - 20
# 绘制刻度
for i in range(60):
    # 以圆心为原点时刻度远端坐标
    xFar = rFar * math.cos(i * math.pi/30)
    yFar = rFar * math.sin(i * math.pi/30)
    # 转换为图像圆心坐标
    xFar_c = int(center_x + xFar)
    yFar_c = int(center_y - yFar)
    # 每5个刻度为粗刻度
    if i % 5 == 0:
        # 以圆心为原点时粗刻度近端坐标
        xNear = rNearBold * math.cos(i * math.pi / 30)
        yNear = rNearBold * math.sin(i * math.pi / 30)
        xNear_c = int(center_x + xNear)
        yNear_c = int(center_y - yNear)
        cv2.line(img, (xFar_c, yFar_c), (xNear_c, yNear_c), color=(0, 0, 0), thickness=9, lineType=cv2.LINE_AA)
    else:
        # 以圆心为原点时细刻度近端坐标
        xNear = rNear * math.cos(i * math.pi / 30)
        yNear = rNear * math.sin(i * math.pi / 30)
        xNear_c = int(center_x + xNear)
        yNear_c = int(center_y - yNear)
        cv2.line(img, (xFar_c, yFar_c), (xNear_c, yNear_c), color=(0, 0, 0), thickness=4, lineType=cv2.LINE_AA)

while True:
    imgCp = img.copy()

    # 绘制日期
    date = datetime.datetime.now().strftime('%m/%d/%Y')
    cv2.putText(imgCp, text=date, org=(center_x - 200, center_y - 200), color=(0, 0, 0),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=2)

    # 绘制时间
    rHour = rFar - 160
    h = datetime.datetime.now().hour % 12
    ind = 0
    if h > 3:
        ind = 15 - h
    else:
        ind = 3 - h
    tickHour_x = rHour * math.cos(ind * math.pi / 6)
    tickHour_y = rHour * math.sin(ind * math.pi / 6)
    tickHour_x_c = int(center_x + tickHour_x)
    tickHour_y_c = int(center_y - tickHour_y)
    cv2.line(imgCp, (center_x, center_y), (tickHour_x_c, tickHour_y_c), color=(255, 125, 0), thickness=10,
             lineType=cv2.LINE_AA)

    rMin = rFar - 100
    rSec = rFar - 40
    minute = datetime.datetime.now().minute
    sec = datetime.datetime.now().second
    indMin = 0
    if minute > 15:
        indMin = 75 - minute
    else:
        indMin = 15 - minute
    indSec = 0
    if sec > 15:
        indSec = 75 - sec
    else:
        indSec = 15 - sec
    tickMin_x = rMin * math.cos(indMin * math.pi / 30)
    tickMin_y = rMin * math.sin(indMin * math.pi / 30)
    tickMin_x_c = int(center_x + tickMin_x)
    tickMin_y_c = int(center_y - tickMin_y)
    cv2.line(imgCp, (center_x, center_y), (tickMin_x_c, tickMin_y_c), color=(255, 125, 0), thickness=7,
             lineType=cv2.LINE_AA)
    tickSec_x = rSec * math.cos(indSec * math.pi / 30)
    tickSec_y = rSec * math.sin(indSec * math.pi / 30)
    tickSec_x_c = int(center_x + tickSec_x)
    tickSec_y_c = int(center_y - tickSec_y)
    cv2.line(imgCp, (center_x, center_y), (tickSec_x_c, tickSec_y_c), color=(255, 125, 0), thickness=2,
             lineType=cv2.LINE_AA)

    cv2.imshow('Clock', imgCp)
    key = cv2.waitKey(5)
    # 按下ESC键退出
    if key == 27:
        break

cv2.destroyAllWindows()


