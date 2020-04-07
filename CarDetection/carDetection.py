import cv2
import numpy as np
from CarDetection.detector import car_detector, bow_features
from CarDetection.pyramid import pyramid
from CarDetection.non_maximum import non_max_suppression_fast as nms
from CarDetection.sliding_window import sliding_window


def in_range(number, test, thresh=0.2):
    return abs(number - test) < thresh


# test_image = r".\resources\cars1.jpg"
test_image = r".\resources\car1.jpg"
# img_path = r".\resources\right-3.jpg"

sift = cv2.xfeatures2d.SIFT_create()
flann_params = dict(algorithm=1, trees=5)
flann_mat = cv2.FlannBasedMatcher(flann_params, {})
svm, bow_extractor = car_detector(40, sift, flann_mat)

w, h = 100, 40
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.imread(test_image, cv2.IMREAD_COLOR)

rectangles = []
counter = 1
scaleFactor = 1.1
scale = 1
font = cv2.FONT_HERSHEY_PLAIN

# 检测时使用图像金字塔检测感兴趣区域
for resized in pyramid(img, scaleFactor):
    scale = float(img.shape[1]) / float(resized.shape[1])
    # 在该层图像金字塔中进行滑动窗口
    for (x, y, roi) in sliding_window(resized, 5, (100, 40)):

        if roi.shape[1] != w or roi.shape[0] != h:
            continue

        try:
            bf = bow_features(roi, bow_extractor, sift)
            _, result = svm.predict(bf)
            _, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            score = res[0][0]
            if result[0][0] == 1.0 and score < -4.0:
                # print("Class: %d, Score: %f" % (result[0][0], score))
                # 计算在原图中窗口位置
                rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale)
                rectangles.append([rx, ry, rx2, ry2, abs(score)])
            # if result[0][0] == -1.0 and score < -1.0:
                # print("Class: %d, Score: %f" % (result[0][0], score))
        except:
            pass

        counter += 1

# 仅在原图上进行滑动窗口
# for (x, y, roi) in sliding_window(img, 5, (w, h)):
#     try:
#         bf = bow_features(roi, bow_extractor, sift)
#         _, result = svm.predict(bf)
#         _, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
#         score = res[0][0]
#         if result[0][0] == 1.0 and score < -1.0:
#             print("Class: %d, Score: %f" % (result[0][0], score))
#             rx, ry, rx2, ry2 = int(x), int(y), int(x + w), int(y + h)
#             rectangles.append([rx, ry, rx2, ry2, abs(score)])
#         if result[0][0] == -1.0 and score < -1.0:
#             print("Class: %d, Score: %f" % (result[0][0], score))
#     except:
#         pass

windows = np.array(rectangles)
print('length of rectangles:', len(rectangles))
print('shape of windows:', windows.shape)  # x * 5
# 再使用NMS方法去除重叠区域
# boxes = nms(windows, 0.15)
# print('length of filtered box:', len(boxes))

for (x, y, x2, y2, score) in windows:
    print(x, y, x2, y2, score)
    cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.putText(img, "%f" % score, (int(x), int(y)), font, 1, (0, 255, 0))

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow("img", img)


cv2.waitKey(0)
cv2.destroyAllWindows()

