import cv2
import numpy as np

# img = cv2.imread(r".\resources\handwriting.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread(r".\resources\j.bmp", cv2.IMREAD_GRAYSCALE)
ret, imgBi = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Origin', cv2.hconcat((img, imgBi)))

# Canny边缘检测器可以直接从原图中查找边缘，边缘不一定连续
edges = cv2.Canny(img, 30, 70)

# 寻找二值化图中的轮廓，仅用于二值化图
# mode参数表示轮廓提取算法
# RETR_EXTERNAL仅仅获取最顶级（即最外部）的轮廓；
# RETR_LIST表示不获取轮廓的从属关系，简单将所有轮廓认为同级，速度最快，但丢失了拓扑结构信息；
# RETR_TREE获取轮廓完整的树状从属关系；
# RETR_CCOMP将轮廓重组织至两个层级，不是外部轮廓就是内部轮廓
# method表示轮廓近似的方法，CHAIN_APPROX_SIMPLE表示尽可能简单地保存，仅保留直线两端的终点
contours, hierarchy = cv2.findContours(imgBi, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
# 返回的第一个值contours即为表示每条轮廓的数组，长度即为轮廓条数，其中每条轮廓即为一组点坐标
print(len(contours))  # 查看轮廓条数
print(len(contours[0]))  # 查看第一条轮廓中的点的数量
print(hierarchy)
print(hierarchy[0])
print(hierarchy[0][0])  # 第1条轮廓的父子关系，四个值分别为[Next, Previous, First Child, Parent]，-1表示不存在
print(hierarchy[0, 0, 0])  # 查看第1条轮廓的下一个同级轮廓的下标(next)

# 在原图上绘制轮廓
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
totImg = img.copy()
totImgHier = img.copy()
totImgFill = img.copy()
totImgFillHier = img.copy()
for i in range(len(contours)):
    # 比较加入层级信息与不加层级信息的轮廓绘制
    imgCp = img.copy()
    cv2.drawContours(imgCp, contours, contourIdx=i, color=(0, 0, 255), thickness=2)
    totImg = cv2.hconcat((totImg, imgCp))
    imgCpHier = img.copy()
    cv2.drawContours(imgCpHier, contours, contourIdx=i, color=(0, 0, 255), thickness=2, hierarchy=hierarchy)
    totImgHier = cv2.hconcat((totImgHier, imgCpHier))

    # 绘制轮廓并填充，比较是否加入层级信息的区别
    imgCpFill = img.copy()
    cv2.drawContours(imgCpFill, contours, contourIdx=i, color=(0, 0, 255), thickness=cv2.FILLED)
    totImgFill = cv2.hconcat((totImgFill, imgCpFill))
    imgCpFillHier = img.copy()
    cv2.drawContours(imgCpFillHier, contours, contourIdx=i, color=(0, 0, 255), thickness=cv2.FILLED, hierarchy=hierarchy)
    totImgFillHier = cv2.hconcat((totImgFillHier, imgCpFillHier))

cv2.imshow('contour in img', cv2.vconcat((totImg, totImgHier)))
cv2.imshow('contour in img Fill', cv2.vconcat((totImgFill, totImgFillHier)))

# contourIdx为负时表示绘制所有轮廓
imgCp = img.copy()
cv2.drawContours(imgCp, contours, contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.imshow('compare contours and canny edges', cv2.hconcat((img, imgCp, edges)))


cv2.waitKey(0)
cv2.destroyAllWindows()

