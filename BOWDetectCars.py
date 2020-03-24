import cv2
import numpy as np

datapath = r".\resources\CarData\TrainImages"


def path(cls, i):
    return "%s\\%s-%d.pgm" % (datapath, cls, i)


pos, neg = "pos", "neg"


# SIFT特征
sift = cv2.xfeatures2d.SIFT_create()
# sift_detect = cv2.xfeatures2d.SIFT_create()
# sift_extract = cv2.xfeatures2d.SIFT_create()

# ORB特征
orb = cv2.ORB_create()
# 使用FLANN特征匹配
flann_params = dict(algorithm=1, tree=5)
flannMat = cv2.FlannBasedMatcher(flann_params, {})

# K均值聚类BOW特征
kmeans_bow_trainer = cv2.BOWKMeansTrainer(80)

# BOW特征提取器
# extract_bow = cv2.BOWImgDescriptorExtractor(orb, flannMat)
extract_bow = cv2.BOWImgDescriptorExtractor(sift, flannMat)


# 提取ORB特征
def extract_orb(fn):
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    kpts, des = orb.detectAndCompute(img, mask=None)
    # kpts, des = orb_extract.compute(img, orb_detect.detect(img))
    return des


# 提取SIFT特征
def extract_sift(fn):
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    kpts, des = sift.detectAndCompute(img, mask=None)
    # kpts, des = sift_extract.compute(img, sift_detect.detect(img))
    return des


# 取前8张图片寻找聚类中心 - “单词”
count_pos = 0
count_neg = 0
totalCount = 8
for i in range(500):
    # des_pos = extract_orb(path(pos, i))
    # des_neg = extract_orb(path(neg, i))
    des_pos = extract_sift(path(pos, i))
    des_neg = extract_sift(path(neg, i))
    if des_pos is not None and len(des_pos) > 0 and count_pos < totalCount:
        kmeans_bow_trainer.add(des_pos)
        print("SIFT des shape:", des_pos.shape)
        count_pos += 1
    if des_neg is not None and len(des_neg) > 0 and count_neg < totalCount:
        kmeans_bow_trainer.add(des_neg)
        count_neg += 1
    if count_pos == totalCount and count_neg == totalCount:
        break

print("Pos count:", count_pos)
print("Neg count:", count_neg)

voc = kmeans_bow_trainer.cluster()
print('Shape of vocabulary:', voc.shape)  # 80 * 128
extract_bow.setVocabulary(voc)


# 从图片获取BOW特征
def bow_feature(fn):
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    return extract_bow.compute(img, sift.detect(img))


# 前20张图片中抽取BOW特征
traindata, trainlabels = [], []
for i in range(100):
    bowDes_pos = bow_feature(path(pos, i))
    # print("BOW Des shape:", bowDes_pos.shape)  # 所有都是 1 * 40 的ndarray
    if bowDes_pos is not None:
        traindata.extend(bowDes_pos)
        trainlabels.append(1)
    bowDes_neg = bow_feature(path(neg, i))
    if bowDes_neg is not None:
        traindata.extend(bowDes_neg)
        trainlabels.append(-1)

traindata = np.array(traindata)
trainlabels = np.array(trainlabels)
print("shape of train data:", traindata.shape)
print("shape of train label:", trainlabels.shape)

# 使用前20张图片中的BOW特征来训练SVM分类器
svm = cv2.ml.SVM_create()
svm.train(traindata, cv2.ml.ROW_SAMPLE, trainlabels)

# 使用训练后的SVM分类器预测新图片
def predict(fn):
    f = bow_feature(fn)
    p = svm.predict(f)
    print(fn, "\t", p[1][0][0])
    return p


car1, car2 = r".\resources\CarData\TestImages\test-1.pgm", r".\resources\car1.jpg"
vr, cup = r".\resources\VR1.png", r".\resources\right-3.jpg"
car1_Img = cv2.imread(car1, cv2.IMREAD_COLOR)
car2_Img = cv2.imread(car2, cv2.IMREAD_COLOR)
vr_Img = cv2.imread(vr, cv2.IMREAD_COLOR)
cup_Img = cv2.imread(cup, cv2.IMREAD_COLOR)

car1_predict = predict(car1)
car2_predict = predict(car2)
vr_predict = predict(vr)
cup_predict = predict(cup)

font = cv2.FONT_HERSHEY_COMPLEX

if car1_predict[1][0][0] == 1.0:
    cv2.putText(car1_Img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
else:
    cv2.putText(car1_Img, 'No Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Car1 Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Car1 Detection', car1_Img)

if car2_predict[1][0][0] == 1.0:
    cv2.putText(car2_Img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
else:
    cv2.putText(car2_Img, 'No Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Car2 Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Car2 Detection', car2_Img)

if vr_predict[1][0][0] == 1.0:
    cv2.putText(vr_Img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
else:
    cv2.putText(vr_Img, 'No Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.namedWindow('VR Detection', cv2.WINDOW_NORMAL)
cv2.imshow('VR Detection', vr_Img)

if cup_predict[1][0][0] == 1.0:
    cv2.putText(cup_Img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
else:
    cv2.putText(cup_Img, 'No Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Cup Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Cup Detection', cup_Img)


cv2.waitKey(0)
cv2.destroyAllWindows()





