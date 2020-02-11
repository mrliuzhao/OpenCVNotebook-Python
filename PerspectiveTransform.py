import cv2
import numpy as np

img = cv2.imread(r".\resources\card.jpg", cv2.IMREAD_UNCHANGED)
rows, cols, channels = img.shape
cv2.imshow('origin', img)

# 卡片的四个角点，非共线的四个点才能唯一确定透视变换
pts1 = np.float32([[148, 80], [437, 114], [94, 247], [423, 288]])
# 变换后分别在左上、右上、左下、右下四个点
pts2 = np.float32([[0, 0], [320, 0], [0, 178], [320, 178]])

# 生成透视变换矩阵，结果为3*3的矩阵
# 前两列组成的3*2矩阵表示2维->3维的变换，最后一列表示原点的平移
M = cv2.getPerspectiveTransform(pts1, pts2)
print(M)
# 进行透视变换
# 2维->3维坐标：X = M11x + M12y + M13; Y = M21x + M22y + M23; Z = M31x + M32y + M33;
# 再将3维坐标映射到2维上: x' = X/Z ; y' = Y/Z  ???
imgPersp = cv2.warpPerspective(img, M, (320, 178))
cv2.imshow('Perspective transform', imgPersp)

# 使用cv2.WARP_INVERSE_MAP flag表示逆向变换
imgPerspInv = cv2.warpPerspective(imgPersp, M, (cols, rows), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
cv2.imshow('Perspective transform Inverse', imgPerspInv)


cv2.waitKey(0)
cv2.destroyAllWindows()

