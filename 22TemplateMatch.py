import cv2
import numpy as np

img = cv2.imread(r".\resources\dog1.jpg", cv2.IMREAD_COLOR)
template = cv2.imread(r".\resources\dog1Face.png", cv2.IMREAD_COLOR)
h, w = template.shape[:2]  # rows->h, cols->w

# 在图片上进行模板匹配。匹配过程类似Valid模式的卷积，即将模板作为核在图片上进行滑动，使用不同方法计算滑动窗口下原图与模板的相似度，具体计算方法有多种：
# TM_SQDIFF方法，原图与模板每个像素的平方差之和，返回的值越小匹配程度越好；
# TM_CCORR方法，原图与模板每个像素的乘积之和，返回值越大表明匹配程度越好；
# TM_CCOEFF方法，原图与模板之间的相关系数，返回值在-1,1之间，越大则该区域越相似
# 三种方法都有对应的归一化方法，即将结果归一化至[0,1]区间
# 由于类似Valid模式的卷积，因此图片大小为(W,H)，模板大小为(w,h)，返回的矩阵大小为(W-w+1, H-h+1)
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)

# 通过minMaxLoc寻找匹配结果矩阵中最大值和最小值，以及其对应的位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# 结果矩阵的最大值位置在对应原图的左上角
left_top = max_loc
right_bottom = (left_top[0] + w, left_top[1] + h)
cv2.rectangle(img, left_top, right_bottom, color=(0, 255, 255), thickness=2)

tempImg = np.zeros(img.shape, np.uint8)
tempImg[:template.shape[0], :template.shape[1], :] = template
cv2.imshow('Image Template Match', cv2.hconcat((img, tempImg)))


cv2.waitKey(0)
cv2.destroyAllWindows()



