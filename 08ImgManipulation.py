import cv2
import numpy as np

img = cv2.imread(r".\resources\cat2.jpg", cv2.IMREAD_UNCHANGED)
rows, cols, channels = img.shape
cv2.imshow('origin', img)

# 按照指定的宽度、高度缩放图片，注意此处的Size为（width,height），与shape返回的次序相反！
shrinkImg = cv2.resize(img, (int(cols/2), int(rows/2)), interpolation=cv2.INTER_AREA)
# 按照比例缩放，如x,y轴均放大一倍
enlargeImg = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
# 关于插值方法（Interpolation），OpenCV官网建议缩小图片时使用cv2.INTER_AREA方法；放大图片时，使用cv2.INTER_CUBIC方法最好，但速度慢，使用cv2.INTER_LINEAR方法速度快，效果还OK

cv2.imshow('shrink', shrinkImg), cv2.imshow('enlarge', enlargeImg)

# 翻转，第2个参数表示翻转方向：
# 0：垂直翻转(沿x轴)；> 0: 水平翻转(沿y轴)；< 0: 水平垂直都翻转(相当于旋转180°)
imgFlip_V = cv2.flip(img, 0)
imgFlip_H = cv2.flip(img, 1)
imgFlip_B = cv2.flip(img, -1)

cv2.imshow('flip vertically', imgFlip_V)
cv2.imshow('flip horizontally', imgFlip_H)
cv2.imshow('flip both', imgFlip_B)

# 沿y=x翻转，相当于逆时针旋转90°后再垂直翻转（或水平翻转后旋转90°）
imgTrans = cv2.transpose(img)
cv2.imshow('transposed', imgTrans)

# 定义平移矩阵，需要是numpy的float32类型
# x轴平移100，y轴平移50的矩阵
# 1  0  100
# 0  1  50
M = np.float32([[1, 0, 100], [0, 1, 50]])
# 用仿射变换实现平移
# 仿射变换矩阵一定是2*3的，前两列即为原点不变的线性变换矩阵，第三列即为变换后原点平移到的位置
# 即dst(x,y)=src(M11x+M12y+M13,M21x+M22y+M23)
imgShift = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('shift', imgShift)

# 绕图片中心逆时针旋转45°，并缩小一半
# 获得对应的放射变换矩阵
# 第一个参数：旋转中心，注意坐标顺序为(x,y)；第二个参数：逆时针旋转角度（负值则表示顺时针）；第三个参数：缩放比例
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
print(M)
imgRot = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('rotation', imgRot)


cv2.waitKey(0)
cv2.destroyAllWindows()

