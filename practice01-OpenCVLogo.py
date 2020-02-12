import cv2
import numpy as np

# 等边三角形边长
b = 110
# 三角形顶点距xy轴边距
a = 50
cols = 2*a + b
rows = int(2*a + 0.866*b)
# 创建一副黑色的图片
img = np.zeros((rows, cols, 3), np.uint8)

# 红色弧线
# cv2.ellipse(img, center=(a+int(b/2), a),
#             axes=(int(0.75*a), int(0.75*a)), angle=120, startAngle=0, endAngle=300,
#             color=(0, 0, 255), thickness=int(a/2), lineType=cv2.LINE_AA)
cv2.ellipse(img, center=(a+int(b/2), a),
            axes=(a, a), angle=120, startAngle=0, endAngle=300,
            color=(0, 0, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
cv2.circle(img, center=(a+int(b/2), a), radius=int(0.4*a),
           color=(0, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

# 绿色弧线
# cv2.ellipse(img, center=(a, a+int(0.866*b)),
#             axes=(int(0.75*a), int(0.75*a)), angle=0, startAngle=0, endAngle=300,
#             color=(0, 255, 0), thickness=int(a/2), lineType=cv2.LINE_AA)
cv2.ellipse(img, center=(a, a+int(0.866*b)),
            axes=(a, a), angle=0, startAngle=0, endAngle=300,
            color=(0, 255, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
cv2.circle(img, center=(a, a+int(0.866*b)), radius=int(0.4*a),
           color=(0, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

# 蓝色弧线
# cv2.ellipse(img, center=(a+b, a+int(0.866*b)),
#             axes=(int(0.75*a), int(0.75*a)), angle=300, startAngle=0, endAngle=300,
#             color=(255, 0, 0), thickness=int(a/2), lineType=cv2.LINE_AA)
cv2.ellipse(img, center=(a+b, a+int(0.866*b)),
            axes=(a, a), angle=300, startAngle=0, endAngle=300,
            color=(255, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
cv2.circle(img, center=(a+b, a+int(0.866*b)), radius=int(0.4*a),
           color=(0, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

# 画黑色三角形
# pts = np.array([[a+int(b/2), a], [a+int(b/4), a+int(0.433*b)],
#                 [a+int(0.75*b), a+int(0.433*b)]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv2.fillPoly(img, pts=[pts], color=(0, 0, 0), lineType=cv2.LINE_AA)
#
# pts = np.array([[a, a+int(0.866*b)], [a+int(b/4), a+int(0.433*b)],
#                 [a+int(b/2), a+int(0.866*b)]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv2.fillPoly(img, pts=[pts], color=(0, 0, 0), lineType=cv2.LINE_AA)
#
# pts = np.array([[a+b, a+int(0.866*b)], [a+int(0.75*b), a+int(0.433*b)],
#                 [int(2*a + 0.866*b), a+int(0.433*b)]], np.int32)
# pts = pts.reshape((-1, 1, 2))
# cv2.fillPoly(img, pts=[pts], color=(0, 0, 0), lineType=cv2.LINE_AA)

cv2.namedWindow('OpenCV Logo', cv2.WINDOW_NORMAL)
cv2.imshow('OpenCV Logo', img)


cv2.waitKey(0)
cv2.destroyAllWindows()

