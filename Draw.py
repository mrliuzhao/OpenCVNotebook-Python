import cv2
import numpy as np

img = cv2.imread(r".\resources\pic2.jpg", cv2.IMREAD_UNCHANGED)
rows, cols, channels = img.shape
cv2.imshow('origin', img)

imgCp = img.copy()
# 沿打开图片的对角线画一条红色直线，宽度5
cv2.line(imgCp, (0, 0), (cols, rows), thickness=5, color=(0, 0, 255), lineType=cv2.LINE_AA)
cv2.line(imgCp, (cols, 0), (0, rows), thickness=5, color=(0, 0, 255), lineType=cv2.LINE_AA)
cv2.imshow('img with lines', imgCp)

# 画一个绿色边框的矩形，参数2：左上角坐标，参数3：右下角坐标
cv2.rectangle(imgCp, (int(cols/4), int(rows/4)), (int(3*cols/4), int(3*rows/4)),
              color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.imshow('add rectangle', imgCp)

# 画一个填充红色的圆，参数2：圆心坐标，参数3：半径
imgCp = img.copy()
cv2.circle(imgCp, (int(cols/2 + 10), int(rows/2 - 30)), 30, color=(0, 0, 255), thickness=cv2.FILLED)
cv2.imshow('add circle', imgCp)

# 画一个椭圆
# 参数2：椭圆中心(x,y)
# 参数3：椭圆(x轴，y轴)的长度
# 参数4：angle — 椭圆的旋转角度，顺时针方向
# 参数5：startAngle — 椭圆的起始角度
# 参数6：endAngle — 椭圆的结束角度
imgCp = img.copy()
cv2.ellipse(imgCp, center=(int(cols/2 + 10), int(rows/2 - 30)),
            axes=(20, 30), angle=-30, startAngle=0, endAngle=360,
            color=(0, 0, 255), thickness=cv2.FILLED)
# 再加一条弧线
cv2.ellipse(imgCp, center=(int(cols/2 + 10), int(rows/2 - 30)),
            axes=(250, 240), angle=-30, startAngle=240, endAngle=320,
            color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
cv2.imshow('add ellipse', imgCp)

# RotatedRect (const Point2f &center, const Size2f &size, float angle)
# img = cv.ellipse(img, box, color[, thickness[, lineType]]


# 画多边形，定义六个顶点坐标
imgCp = img.copy()
center_x = int(cols/2)
center_y = int(rows/2)
pts = np.array([[center_x, center_y - 60], [center_x + 52, center_y - 30],
                [center_x + 52, center_y + 30], [center_x, center_y + 60],
                [center_x - 52, center_y + 30], [center_x - 52, center_y - 30]], np.int32)
# 顶点个数：6，矩阵变成6*1*2维
pts = pts.reshape((-1, 1, 2))
cv2.polylines(imgCp, pts=[pts], isClosed=True, color=(0, 255, 255), thickness=5, lineType=cv2.LINE_AA)
# 添加文字，起点位置org为左下角
cv2.putText(imgCp, text='I Love you!', org=(center_x - 150, center_y - 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
cv2.imshow('add hexagon', imgCp)


cv2.waitKey(0)
cv2.destroyAllWindows()
