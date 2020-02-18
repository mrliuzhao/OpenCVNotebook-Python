import cv2
import numpy as np

img = cv2.imread(r".\resources\j.bmp", cv2.IMREAD_COLOR)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, imgBi = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 查找轮廓
contours, hierarchy = cv2.findContours(imgBi, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# 分别绘制轮廓线
totImg = img.copy()
for i in range(len(contours)):
    # 比较加入层级信息与不加层级信息的轮廓绘制
    imgCp = img.copy()
    cv2.drawContours(imgCp, contours, contourIdx=i, color=(0, 0, 255), thickness=2)
    totImg = cv2.hconcat((totImg, imgCp))

cv2.imshow('Contours', totImg)

area = []
perimeter = []
totImg = img.copy()
for i in range(len(contours)):
    # 计算轮廓面积
    area.append(cv2.contourArea(contours[i]))
    print('第{}条轮廓下的面积为：{}'.format((i+1), area[i]))

    # 计算轮廓周长
    perimeter.append(cv2.arcLength(contours[i], True))
    print('第{}条轮廓下的周长为：{}'.format((i+1), perimeter[i]))

    # 计算轮廓的几何中心矩，即 m_{pq} = \Sigma_i \Sigma_j i^p j^q f(i,j)
    M = cv2.moments(contours[i])
    # m00即为面积，即为区域内所有灰度值的加和
    print('第{}条轮廓下的矩：m00 = {}; m10 = {}; m01 = {}; m20 = {}; m11 = {}; m02 = {}; '
          'm30 = {}; m21 = {}; m12 = {}; m03 = {}'
          .format((i+1), M['m00'], M['m10'], M['m01'], M['m20'], M['m11'], M['m02'],
                  M['m30'], M['m21'], M['m12'], M['m03']))
    # 利用矩计算轮廓的几何质心
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    print('第{}条轮廓的几何质心：({},{})'.format((i+1), cx, cy))
    # 计算轮廓的Hu矩
    huM = cv2.HuMoments(M)
    print('第{}条轮廓下的Hu矩：Hu1 = {}; Hu2 = {}; Hu3 = {}; Hu4 = {}; Hu5 = {}; Hu6 = {}; Hu7 = {}'
          .format((i+1), huM[0], huM[1], huM[2], huM[3], huM[4], huM[5], huM[6]))

# 统计所有非零像素点个数，一定是整数
count = cv2.countNonZero(imgBi)
print(count)  # 2576
print(area[0] + area[3])  # 2590.0 -- 更贴近count，貌似计算面积仅仅计算了白色部分（前景）的面积
print(area[0] + area[3] - area[1])  # 2307.0

# 考察轮廓的外接矩形
totImg = img.copy()
for i in range(len(contours)):
    # 外接矩形，一定是垂直的
    imgCp = img.copy()
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(imgCp, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 最小外接矩形，可能带有旋转，返回的是RotatedRectangle
    rect = cv2.minAreaRect(contours[i])
    # 获取旋转矩形的四个角点，并取整
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(imgCp, [box], 0, (255, 0, 0), 2)

    # 把几何质心也画出来
    M = cv2.moments(contours[i])
    # 利用矩计算轮廓的几何质心
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    cv2.circle(imgCp, center=(int(cx), int(cy)), radius=2, color=(0, 0, 255), thickness=cv2.FILLED)
    totImg = cv2.hconcat((totImg, imgCp))

cv2.imshow('Bounding Rectangle', totImg)


# 考察轮廓的外接圆
totImg = img.copy()
for i in range(len(contours)):
    # 外接圆 - 红色
    imgCp = img.copy()
    (x, y), radius = cv2.minEnclosingCircle(contours[i])
    (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
    cv2.circle(imgCp, (x, y), radius, (0, 0, 255), 2)

    # 拟合椭圆，方法返回的是一个RotatedRectangle，拟合的椭圆即为其内切椭圆
    # 青色
    ellipse = cv2.fitEllipse(contours[i])
    cv2.ellipse(imgCp, ellipse, (255, 255, 0), 2)

    # 拟合椭圆的另外两种方法
    # 蓝色
    ellipse = cv2.fitEllipseAMS(contours[i])
    cv2.ellipse(imgCp, ellipse, (255, 0, 0), 2)
    # 绿色
    ellipse = cv2.fitEllipseDirect(contours[i])
    cv2.ellipse(imgCp, ellipse, (0, 255, 0), 2)
    totImg = cv2.hconcat((totImg, imgCp))

cv2.imshow('Bounding Circle', totImg)


# 考察轮廓的外接多边形
totImg = img.copy()
for i in range(len(contours)):
    # 多边形逼近，得到多边形的角点。使用Douglas-Peucker算法
    imgCp = img.copy()
    approx = cv2.approxPolyDP(contours[i], epsilon=3, closed=True)
    cv2.polylines(imgCp, [approx], True, (0, 255, 0), 2)
    print('第{}条轮廓点数 : 多边形逼近的点数 = {} : {}'.format((i+1), len(contours[i]), len(approx)))

    # 判断形状是否是凸的
    isConvex = cv2.isContourConvex(contours[i])
    print('第{}条轮廓是否凸？{}'.format((i+1), isConvex))
    # 绘制凸包
    hull = cv2.convexHull(contours[i])
    cv2.polylines(imgCp, [hull], True, (0, 0, 255), 2)

    totImg = cv2.hconcat((totImg, imgCp))

cv2.imshow('Bounding Rectangle', totImg)

# 检测两个形状之间的相似度，返回值越小，越相似
# 其具体算法就是通过Hu矩进行衡量的
print('0-1之间相似度（方法1）：', cv2.matchShapes(contours[0], contours[1], cv2.CONTOURS_MATCH_I1, 0.0))
print('0-1之间相似度（方法2）：', cv2.matchShapes(contours[0], contours[1], cv2.CONTOURS_MATCH_I2, 0.0))
print('0-1之间相似度（方法3）：', cv2.matchShapes(contours[0], contours[1], cv2.CONTOURS_MATCH_I3, 0.0))
print('1-3之间相似度（方法1）：', cv2.matchShapes(contours[1], contours[3], cv2.CONTOURS_MATCH_I1, 0.0))
print('1-3之间相似度（方法2）：', cv2.matchShapes(contours[1], contours[3], cv2.CONTOURS_MATCH_I2, 0.0))
print('1-3之间相似度（方法3）：', cv2.matchShapes(contours[1], contours[3], cv2.CONTOURS_MATCH_I3, 0.0))


cv2.waitKey(0)
cv2.destroyAllWindows()

