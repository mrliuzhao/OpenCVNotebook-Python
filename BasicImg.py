import cv2
import numpy as np

picMat = cv2.imread(r".\resources\pic2.jpg", cv2.IMREAD_COLOR)
print("picMat is null ", (picMat is None))
if picMat is None:
    print("Error: Cannot load image!")
    exit(-1)

cv2.imshow("Picshow", picMat)

print("Type of pic is ", type(picMat))  # Type of pic is  <class 'numpy.ndarray'>
print("shape:", picMat.shape)  # shape: (1000, 520, 3)  Height(y), Width(x), Channels
print("strides:", picMat.strides)  # strides: (1560, 3, 1)
print("ndim:", picMat.ndim)  # ndim: 3
print("size:", picMat.size)  # size: 1560000
print("dtype:", picMat.dtype)  # dtype: uint8

height, width, channel = picMat.shape
# 高度减半
picHalf = picMat[0:int(height/2):, :, :]
cv2.imshow("PicshowHalf", picHalf)
# 水平翻转
picInv_H = picMat[::, ::-1, :]
cv2.imshow("Horizontally Inverse", picInv_H)
# 垂直翻转
picInv_V = picMat[::-1, :, :]
cv2.imshow("Vertically Inverse", picInv_V)
# 旋转180度，并简单降采样
picInv = picMat[::-2, ::-2, :]
cv2.imshow("Totally Inverse", picInv)

# 通道分离
picBlue = picMat.copy()
picBlue[:, :, 1] = 0
picBlue[:, :, 2] = 0
cv2.imshow("PicshowBlue", picBlue)

picGreen = picMat.copy()
picGreen[:, :, 0] = np.zeros((height, width), 'uint8')
picGreen[:, :, 2] = np.zeros((height, width), 'uint8')
cv2.imshow("PicshowGreen", picGreen)

# split方法效率不够高
b, g, r = cv2.split(picMat)
picRed = cv2.merge((np.zeros((height, width), 'uint8'), np.zeros((height, width), 'uint8'), r))
cv2.imshow("PicshowRed", picRed)

# 通道合并，绕y=x翻转
picTrans = cv2.merge((picMat[:, :, 0].T, picMat[:, :, 1].T, picMat[:, :, 2].T))
cv2.imshow("PicshowTranspose", picTrans)

# 从BGR -> RGB
picMatInv = picMat[::, ::, ::-1]
cv2.imshow("PicshowInverse", picMatInv)


cv2.waitKey(0)
cv2.destroyAllWindows()

