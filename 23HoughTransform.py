import cv2
import numpy as np


img = np.zeros((500, 500), np.uint8)
cv2.rectangle(img, pt1=(100, 140), pt2=(100+100, 140+50), color=255, thickness=cv2.FILLED)
cv2.circle(img, center=(300, 350), radius=88, color=255, thickness=cv2.FILLED)
# 一条平行于x轴的线（theta = 0），距离原点rho=33
cv2.line(img, pt1=(200, 33), pt2=(450, 33), color=255, thickness=5)
# 一条直线，距离原点约333.95
cv2.line(img, pt1=(60, 430), pt2=(388, 70), color=255, thickness=2, lineType=cv2.LINE_AA)
cv2.imshow('Origin', img)
# 应该仅对提取出的边缘进行直线检测，否则结果很多
edges = cv2.Canny(img, 50, 150)
cv2.imshow('edges', edges)


# 使用霍夫变换（Multi-scale Hough Transform）在二值化图像中寻找直线
# rho参数表示检测直线到原点的距离的精度，以像素为单位，越小检测直线越多；
# theta参数表示检测直线与y轴负方向（向上）夹角的弧度精度（以顺时针为正），以弧度为单位，越小检测直线越多；
# threshold累加计数阈值，累加值超过该值的直线才会检测到，越大检测到的直线越少；
# srn和stn是多尺度霍夫变换的参数，默认值为0，即使用传统霍夫变换
# min_theta和max_theta即为检测直线与y轴负向夹角的弧度范围
# 返回检测到的直线，每个元素为2维数组，(rho, theta)，含义与参数中一致，不同直线之间的差值也一定是参数的倍数，体现了精度
lines = cv2.HoughLines(edges, rho=0.8, theta=np.pi/180, threshold=90)
print('find {} lines'.format(lines.shape[0]))

imgCol = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
imgCp = imgCol.copy()
h, w = imgCp.shape[:2]
# 将检测的线画出来（注意需要将极坐标转换为直角坐标）
# 对应直线为  y = -cot(theta) x + rho/sin(theta)
for line in lines:
    rho, theta = line[0]
    pts = []
    if theta == 0:  # 平行于y轴
        pts.append((rho, 0))
        pts.append((rho, 500))
        print('平行于y轴的线，rho={}'.format(rho))
    elif theta == np.pi / 2:  # 平行于x轴
        pts.append((0, rho))
        pts.append((500, rho))
        print('平行于x轴的线，rho={}'.format(rho))
    else:
        a = -1.0 / np.tan(theta)
        b = rho / np.sin(theta)
        x0 = rho / np.cos(theta)  # (x0, 0)
        y0 = rho / np.sin(theta)  # (0, y0)
        x500 = (h - b) / a  # (x500, 500)
        y500 = a * w + b  # (500, y500)
        if 0 < x0 < w:
            pts.append((x0, 0))
        if 0 < y0 < h:
            pts.append((0, y0))
        if 0 < y500 < h:
            pts.append((w, y500))
        if 0 < x500 < w:
            pts.append((x500, h))

        if len(pts) != 2:
            print('计算出不符合要求的直线')
        else:
            cv2.line(imgCp, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), (0, 0, 255))
cv2.imshow('Find Line by Hough', cv2.hconcat((imgCol, imgCp)))

# 使用升级版霍夫变换--统计概率霍夫线变换检测直线，可以检测到端点
# rho、theta和threshold与标准霍夫变换中的意义一致
# minLineLength检测直线的最短长度，短于该长度的直线忽略
# maxLineGap检测直线的最大间隔，小于该间隔的认为同一条直线
# 返回检测到的直线，每个元素为4维数组，(x1, y1, x2, y2)分别为直线的两个端点
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=50, maxLineGap=10)
print('find {} lines'.format(lines.shape[0]))
imgCp = imgCol.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(imgCp, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
cv2.imshow('Find Line by HoughP', cv2.hconcat((imgCol, imgCp)))


# 霍夫圆变换检测圆形
# method使用的霍夫变换方法，目前仅实现了HOUGH_GRADIENT一个；
# dp表示计数器与原图大小的比例，1表示计数器与原图一样大，2表示计数器宽和高均变为原图1/2
# minDist表示检测圆心的最短距离，过小会有很多假阳性，过大又会miss一些圆
# param1在HOUGH_GRADIENT方法中，该参数表示Canny边缘检测的高阈值（低阈值直接取一半）
# param2在HOUGH_GRADIENT方法中，该参数即为计数器的阈值，同直线检测中的threshold参数
# minRadius和maxRadius即为检测圆形的半径范围
# 返回检测到的圆形，每个元素为3维数组，(x, y, radius)分别为圆心的原点和半径
circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=100, param2=30)
print('find {} circles'.format(circles[0].shape[0]))
print(circles[0][0])
print(circles[0][1])
imgCp = imgCol.copy()
for circle in circles[0]:
    x, y, r = np.int0(np.around(circle))
    cv2.circle(imgCp, (x, y), r, (0, 0, 255), thickness=2)
    cv2.circle(imgCp, (x, y), 2, (0, 255, 0), thickness=2)
cv2.imshow('Find Circle by Hough', cv2.hconcat((imgCol, imgCp)))


cv2.waitKey(0)
cv2.destroyAllWindows()

