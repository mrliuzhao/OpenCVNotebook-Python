import cv2
import matplotlib.pyplot as plt

# 灰度图方式读入图片
img = cv2.imread(r".\resources\gradualGray.jpg", cv2.IMREAD_GRAYSCALE)

# 应用5种不同的阈值方法
# 直接简单二值阈值分割（大于阈值时取第三个参数的值，小于阈值取0），提取高亮部分（但细节丢失）
ret, thBi = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# 简单二值阈值分割后取反，即大于阈值时取0，小于阈值时取第三个参数的值，提取黑暗部分（同样细节丢失）
ret, thBiInv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# 直接抹平所有超过阈值的部分，仅抹平到阈值，即高亮部分丢失所有细节
ret, thTrunk = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# 未超过阈值的直接取0，超过阈值的保留原值，即仅保留高亮部分所有细节
ret, th20 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# 超过阈值的直接取0，未超过阈值的保留原值，即仅保留低于阈值部分所有细节，直接抹黑高亮，类似Trunk，但处理高亮部分比trunk更厉害
ret, th20Inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thBi, thBiInv, thTrunk, th20, th20Inv]

# 使用Matplotlib显示
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴

plt.show()

# 自适应阈值对比固定阈值
img = cv2.imread(r".\resources\sudoku.jpg", 0)

# 固定阈值
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# 自适应阈值，仅用于BINARY或BINARY_INV两种阈值类型（即严格的“二值化”）
# 计算11*11方块内的均值，再减去4（最后一个参数）作为阈值
th2 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
# 计算17*17方块内的高斯均值，再减去6（最后一个参数）作为阈值
th3 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)

titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
