import cv2
import numpy as np

img = cv2.imread(r".\resources\cat2.jpg", cv2.IMREAD_UNCHANGED)
rows, cols, channels = img.shape
cv2.imshow('origin', img)

# 通过变换前后的三个点坐标获取对应的仿射变换矩阵
# 变换前的三个点
pts1 = np.float32([[0, 0], [1, 0], [0, 1]])
# 变换后的三个点，逆时针旋转90°，均下移宽度位置
pts2 = np.float32([[0, cols], [0, -1+cols], [1, cols]])

# 生成变换矩阵
M = cv2.getAffineTransform(pts1, pts2)
print(M)
imgRot = cv2.warpAffine(img, M, (rows, cols))
cv2.imshow('image Rotation', imgRot)

# 绕原点逆时针旋转90°
M = cv2.getRotationMatrix2D((0, 0), 90, 1)
# 旋转后，原点需要下移宽度位置
M[:, 2] = [0, cols]
print(M)
imgRot = cv2.warpAffine(img, M, (rows, cols))
cv2.imshow('rotation', imgRot)

# 绕图片中心逆时针旋转90°，则原点需要再移动(h/2-w/2, w/2-h/2)
# M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
# M += np.float32([[0, 0, (rows-cols)/2], [0, 0, (cols-rows)/2]])
# print(M)
# imgRot2 = cv2.warpAffine(img, M, (rows, cols))
# cv2.imshow('rotation2', imgRot2)

# 比较带有插值的缩小和简单降频采样的缩小
shrinkImg = cv2.resize(img, (int(cols/4), int(rows/4)), interpolation=cv2.INTER_AREA)
shrinkImgSelf = img[::4, ::4, :]
cv2.namedWindow("shrink Officially", cv2.WINDOW_NORMAL)
cv2.imshow('shrink Officially', shrinkImg)
cv2.namedWindow("shrink Simply", cv2.WINDOW_NORMAL)
cv2.imshow("shrink Simply", shrinkImgSelf)
# 比较带有插值的放大和简单放大
enlargeImg = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
M = np.float32([[4, 0, 0], [0, 4, 0]])
enlargeSelf = cv2.warpAffine(img, M, (cols*4, rows*4))
cv2.imshow('enlarge Officially', enlargeImg)
cv2.imshow("enlarge Simply", enlargeSelf)


cv2.waitKey(0)
cv2.destroyAllWindows()

