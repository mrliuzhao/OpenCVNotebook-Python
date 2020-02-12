import cv2

drawing = False  # 是否在画图
startPos = (-1, -1)
mode = True  # True为画矩形；FALSE为画圆形

img = cv2.imread(r".\resources\pic3.jpg", cv2.IMREAD_COLOR)
rows, cols, channels = img.shape
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
imgCp = img.copy()


# 鼠标的回调函数
def mouse_event(event, x, y, flags, param):
    global drawing, startPos, mode, imgCp, img

    # 左键按下则开始画图
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        startPos = (x, y)
        print('按下左键，开始做图')
    # 鼠标移动，画图
    elif event == cv2.EVENT_MOUSEMOVE:
        imgCp = img.copy()
        cv2.putText(imgCp, '({0},{1})'.format(x, y), org=(0, 100), color=(0, 255, 255),
                    thickness=5, fontScale=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        if drawing:
            if mode:
                cv2.rectangle(imgCp, startPos, (x, y), color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
            else:
                cv2.circle(imgCp, (x, y), 50, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        print('抬起左键，停止作图')
        if mode:
            cv2.rectangle(img, startPos, (x, y), color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), 50, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    # elif event == cv2.EVENT_RBUTTONDBLCLK:
    #     print('双击右键，鼠标位置：', (x, y))


# 定义鼠标的回调函数
cv2.setMouseCallback('image', mouse_event)

while True:
    cv2.imshow('image', imgCp)
    key = cv2.waitKey(5)
    # 按下ESC键退出
    if key == 27:
        break
    # 按下m切换模式
    elif key == ord('m'):
        mode = not mode


cv2.destroyAllWindows()
