import cv2

img1 = cv2.imread(r".\resources\Luffy1.jpeg", cv2.IMREAD_COLOR)
img2 = cv2.imread(r".\resources\cartoon1.jpg", cv2.IMREAD_COLOR)

img = cv2.hconcat((img1, img2))
cv2.imshow('Origin Pic', img)

# 简单叠加两张图片
imgAdd = cv2.add(img1, img2)
cv2.imshow('Simple Add', imgAdd)

# 加权叠加
imgAddW = cv2.addWeighted(img1, 0.7, img2, 0.3, -40)
cv2.imshow('Weighted Add', imgAddW)

# 实现抠图叠加
# 先取出关心区域ROI，路飞的手掌部分
rows1, cols1 = img1.shape[:2]
rows2, cols2 = img2.shape[:2]
roi = img1[366:494, 792:972]
# 对img2降采样
img2Shrink = cv2.resize(img2, None, fx=1/6, fy=1/6, interpolation=cv2.INTER_CUBIC)
cv2.imshow('shinked img2', img2Shrink)

# 创建掩膜
img2gray = cv2.cvtColor(img2Shrink, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# 保留图2有颜色的部分
img2Content = cv2.bitwise_and(img2Shrink, img2Shrink, mask=mask_inv)
cv2.imshow('Picked Part in img2', img2Content)
# 删除ROI有颜色的部分
roiRem = cv2.bitwise_and(roi, roi, mask=mask)
cv2.imshow('Picked Part in ROI', roiRem)
roiAdd = cv2.add(roiRem, img2Content)  # 进行融合
cv2.imshow('Picked Add', roiAdd)
img1[366:494, 792:972] = roiAdd  # 融合后放回原图
cv2.imshow('Picked Add in img1', img1)


cv2.waitKey(0)
cv2.destroyAllWindows()
