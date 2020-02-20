import cv2
import numpy as np

# img = cv2.imread(r".\resources\noisy.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imshow('origin', img)
# print('Origin shape:', img.shape)
#
# # 固定值padding边框，上下左右均添加一个border，统一都填充0也称为zero padding
# imgPad = cv2.copyMakeBorder(img, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_CONSTANT, value=0)
# cv2.imshow('Padding zeros', imgPad)
# print('Padded shape:', imgPad.shape)
#
# # 平均卷积核
# kernel = np.ones((17, 17), np.float32) / 49
# print(kernel)
# unifFilter = cv2.filter2D(img, -1, kernel)
# cv2.imshow('unif', unifFilter)
# print('unif shape:', unifFilter.shape)
#
# # OpenCV自带的blur函数就是均值滤波
# blur = cv2.blur(img, (9, 9))
# cv2.imshow('blur', blur)
# print('blur shape:', blur.shape)
#
# # 高斯滤波器
# gaussian = cv2.GaussianBlur(img, (9, 9), 3)
# cv2.imshow('gaussian blur', gaussian)
# print('gaussian blur shape:', gaussian.shape)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

os = np.zeros((5, 5), np.uint8)
print(os)
os[2, 2] = 1
print(os)

arr = np.array([[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]], np.int32)

res = np.where(arr > 4)
print(res)
for pt in zip(*res[::-1]):
    print(pt)


# 1. 霍夫直线变换
img = cv2.imread(r".\resources\shapes.jpg")
drawing = np.zeros(img.shape[:], dtype=np.uint8)  # 创建画板
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 霍夫直线变换
lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)

# 将检测的线画出来（注意是极坐标噢）
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(drawing, (x1, y1), (x2, y2), (0, 0, 255))

cv2.imshow('hough lines', np.hstack((img, drawing)))
cv2.waitKey(0)
