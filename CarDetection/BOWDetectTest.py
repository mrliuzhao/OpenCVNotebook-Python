import cv2
import numpy as np
from CarDetection.detector import car_detector, bow_features
from CarDetection.pyramid import pyramid
from CarDetection.non_maximum import non_max_suppression_fast as nms
from CarDetection.sliding_window import sliding_window
import time


def in_range(number, test, thresh=0.2):
    return abs(number - test) < thresh


# test_image = r".\resources\cars1.jpg"
# test_image = r".\resources\car1.jpg"
test_image = r".\resources\CarData\TestImages_Scale\test-12.pgm"
# test_image = r".\resources\right-3.jpg"

sift = cv2.xfeatures2d.SIFT_create()
flann_params = dict(algorithm=1, trees=5)
flann_mat = cv2.FlannBasedMatcher(flann_params, {})

svm = None
vocabulary = None

try:
    svm = cv2.ml.SVM_load(r".\resources\carBOWSVM.xml")
except Exception as e:
    print("Cannot Load SVM from file")

try:
    vocabulary = np.load(r".\resources\carSIFT200.npy")
    print("vocabulary shape:", vocabulary.shape)
except Exception as e:
    print("Cannot Load vocabulary from file")

cluster_count = 200
if svm is None or vocabulary is None:
    svm, bow_extractor, vocabulary = car_detector(cluster_count, sift, flann_mat)
    svm.save(r".\resources\carBOWSVM.xml")
    np.save(r".\resources\carSIFT200.npy", vocabulary)
else:
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann_mat)
    bow_extractor.setVocabulary(vocabulary)

# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.imread(test_image, cv2.IMREAD_COLOR)
# img = cv2.resize(img, dsize=(0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
print('img shape:', img.shape)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rectangles = []
counter = 0
scaleFactor = 1.3
scale = 1
font = cv2.FONT_HERSHEY_PLAIN
imgh, imgw = img.shape[:2]
w, h = int(imgw/10), int(imgh/10)

# 检测时使用图像金字塔检测感兴趣区域
start = time.time()
nonCount = 0
bfcount = 0
for resized in pyramid(img_gray, scaleFactor, min_size=(w, h)):
    scale = float(img.shape[1]) / float(resized.shape[1])
    print('scale is', scale)
    # 在该层图像金字塔中进行滑动窗口
    for (x, y, roi) in sliding_window(resized, 5, (w, h)):
        if roi.shape[1] != w or roi.shape[0] != h:
            continue

        try:
            bf = bow_features(roi, bow_extractor, sift)
            if bf is not None:
                bfcount += 1
            else:
                nonCount += 1
            # if bf is not None:
            #     print('bf shape:', bf.shape)  # (1, cluster_count)
            #     print('bf dtype:', bf.dtype)  # float32
            _, result = svm.predict(bf)
            _, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            score = res[0][0]
            if result[0][0] == 1.0 and score < -3.0:
                # print("Class: %d, Score: %f" % (result[0][0], score))
                # 计算在原图中窗口位置
                rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale)
                rectangles.append([rx, ry, rx2, ry2, abs(score)])
            # if result[0][0] == -1.0 and score < -1.0:
                # print("Class: %d, Score: %f" % (result[0][0], score))
        except:
            pass
        counter += 1
end = time.time()
print("金字塔滑动窗口检测时间:", (end - start))  # 42.439358711242676
print("循环总次数：", counter)  # 21999
print("不为空的BOW特征总数：", bfcount)  #
print("为空的BOW特征总数：", nonCount)  #

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
start = time.time()
boxes = nms(windows, 0.6)
# boxes = rectangles
print('length of filtered box:', len(boxes))
end = time.time()
print("NMS方法去重时间:", (end - start))  # 0.0

for (x, y, x2, y2, score) in boxes:
    # print(x, y, x2, y2, score)
    cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.putText(img, "%f" % score, (int(x), int(y)), font, 1, (0, 255, 0))

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow("img", img)


cv2.waitKey(0)
cv2.destroyAllWindows()

