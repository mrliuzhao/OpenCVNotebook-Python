import cv2

img = cv2.imread(r'./resources/circle_ring.jpg', cv2.IMREAD_COLOR)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
ret, imgBi = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 寻找轮廓
contours, hierarchy = cv2.findContours(imgBi, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
print(len(hierarchy[0]))
print(hierarchy)

imgCp = img.copy()
for i in range(len(contours)):
    # 仅填充第二级轮廓内容，即有父节点并且没有子节点的轮廓
    if hierarchy[0, i, 2] == -1 and hierarchy[0, i, 3] != -1:
        cv2.drawContours(imgCp, contours, contourIdx=i, color=(0, 255, 255), thickness=cv2.FILLED)
cv2.imshow('Answer', imgCp)

cv2.waitKey(0)
cv2.destroyAllWindows()

