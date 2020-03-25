import cv2
import numpy as np

'''
该文件用于使用UIUC数据集训练出识别汽车的BOW+SVM模型
'''

datapath = r".\resources\CarData\TrainImages"
SAMPLES = 400


def path(cls, i):
    return "%s/%s%d.pgm" % (datapath, cls, i)


def get_flann_matcher():
    flann_params = dict(algorithm=1, trees=5)
    return cv2.FlannBasedMatcher(flann_params, {})


def get_bow_extractor(extract, match):
    return cv2.BOWImgDescriptorExtractor(extract, match)


def get_extract_detect():
    return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()


def extract_sift(fn, extractor):
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    kpts, des = extractor.detectAndCompute(img, mask=None)
    return des


def bow_features(img, extractor_bow, detector):
    return extractor_bow.compute(img, detector.detect(img))


def car_detector(cluster_count=40,
                 extractor=cv2.xfeatures2d.SIFT_create(),
                 matcher=cv2.FlannBasedMatcher()):
    '''
    该函数用于获取识别汽车的SVM分类器，以及BOW特征提取器

    :param cluster_count: 聚类个数，即词袋中单词种类数
    :param extractor: 特征提取器，如ORB、SIFT、SURF等
    :param matcher: 特征匹配器，如FLANNMatcher
    :return: 第一个返回值为SVM分类器，第二个返回值为BOW特征提取器
    '''

    pos, neg = "pos-", "neg-"
    print("building BOWKMeansTrainer...")
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(cluster_count)
    extract_bow = cv2.BOWImgDescriptorExtractor(extractor, matcher)

    print("adding features to trainer")
    for i in range(SAMPLES):
        kpts, sift_pos = extractor.detectAndCompute(cv2.imread(path(pos, i), cv2.IMREAD_GRAYSCALE), mask=None)
        if sift_pos is not None:
            bow_kmeans_trainer.add(sift_pos)
        kpts, sift_neg = extractor.detectAndCompute(cv2.imread(path(neg, i), cv2.IMREAD_GRAYSCALE), mask=None)
        if sift_neg is not None:
            bow_kmeans_trainer.add(sift_neg)

    vocabulary = bow_kmeans_trainer.cluster()
    extract_bow.setVocabulary(vocabulary)

    traindata, trainlabels = [], []
    print("adding to train data")
    for i in range(SAMPLES):
        # print(i)
        bowDes_pos = bow_features(cv2.imread(path(pos, i), cv2.IMREAD_GRAYSCALE), extract_bow, extractor)
        if bowDes_pos is not None:
            traindata.extend(bowDes_pos)
            trainlabels.append(1)
        bowDes_neg = bow_features(cv2.imread(path(neg, i), cv2.IMREAD_GRAYSCALE), extract_bow, extractor)
        if bowDes_neg is not None:
            traindata.extend(bowDes_neg)
            trainlabels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(1)
    svm.setC(35)
    svm.setKernel(cv2.ml.SVM_RBF)

    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
    return svm, extract_bow
