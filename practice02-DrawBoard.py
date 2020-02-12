import cv2
import numpy as np

drawing = False  # 是否在画图

img = np.zeros((1024, 1024, 3), np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
imgCp = img.copy()
brushsize = 5


def nothing(x):
    pass


# 鼠标的回调函数
def mouse_event(event, x, y, flags, param):
    global drawing, imgCp, img, brushsize
    # 获取滑块的值
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    brushsize = cv2.getTrackbarPos('BrushSize', 'image')

    # 左键按下则开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        print('按下左键，开始做图')
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        imgCp = img.copy()
        cv2.putText(imgCp, '({0},{1})'.format(x, y), org=(0, 30), color=(0, 255, 255),
                    thickness=2, fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        if drawing:
            cv2.circle(imgCp, (x, y), brushsize, color=(b, g, r), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(img, (x, y), brushsize, color=(b, g, r), thickness=-1, lineType=cv2.LINE_AA)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        print('抬起左键，停止作图')


# 定义鼠标的回调函数
cv2.setMouseCallback('image', mouse_event)
# 创建RGB三个滑动条
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
# 创建笔刷大小滑动条
cv2.createTrackbar('BrushSize', 'image', 5, 20, nothing)

while True:
    cv2.imshow('image', imgCp)
    key = cv2.waitKey(1)
    # 按下ESC键退出
    if key == 27:
        break

cv2.destroyAllWindows()
