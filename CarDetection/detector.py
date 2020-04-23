import cv2
import numpy as np
import time

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
    start = time.time()
    for i in range(SAMPLES):
        kpts, sift_pos = extractor.detectAndCompute(cv2.imread(path(pos, i), cv2.IMREAD_GRAYSCALE), mask=None)
        if sift_pos is not None:
            bow_kmeans_trainer.add(sift_pos)
        kpts, sift_neg = extractor.detectAndCompute(cv2.imread(path(neg, i), cv2.IMREAD_GRAYSCALE), mask=None)
        if sift_neg is not None:
            bow_kmeans_trainer.add(sift_neg)

    vocabulary = bow_kmeans_trainer.cluster()
    print("Vocabulary Shape:", vocabulary.shape)  # (cluster_count, 128)
    extract_bow.setVocabulary(vocabulary)
    end = time.time()
    print("训练BOW时间：", (end - start))

    traindata, trainlabels = [], []
    print("adding to train data")
    start = time.time()
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
    end = time.time()
    print("训练SVM时间：", (end - start))

    return svm, extract_bow, vocabulary


def train_bowextractor(cluster_count=40,
                       extractor=cv2.xfeatures2d.SIFT_create(),
                       matcher=cv2.FlannBasedMatcher()):
    '''
    该函数用于训练BOW特征提取器

    :param cluster_count: 聚类个数，即词袋中单词种类数
    :param extractor: 特征提取器，如ORB、SIFT、SURF等
    :param matcher: 特征匹配器，如FLANNMatcher
    :return: 第一个返回值为“视觉单词”（K均值聚类出的中心），第二个返回值为BOW特征提取器
    '''

    pos, neg = "pos-", "neg-"
    print("building BOWKMeansTrainer...")
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(cluster_count)
    extract_bow = cv2.BOWImgDescriptorExtractor(extractor, matcher)

    print("adding features to bow k-means trainer")
    start = time.time()
    for i in range(SAMPLES):
        kpts, sift_pos = extractor.detectAndCompute(cv2.imread(path(pos, i), cv2.IMREAD_GRAYSCALE), mask=None)
        if sift_pos is not None:
            bow_kmeans_trainer.add(sift_pos)
        kpts, sift_neg = extractor.detectAndCompute(cv2.imread(path(neg, i), cv2.IMREAD_GRAYSCALE), mask=None)
        if sift_neg is not None:
            bow_kmeans_trainer.add(sift_neg)

    vocabulary = bow_kmeans_trainer.cluster()
    print("Vocabulary Shape:", vocabulary.shape)  # (cluster_count, 128)
    extract_bow.setVocabulary(vocabulary)
    end = time.time()
    print("训练BOW时间：", (end - start))

    return vocabulary, extract_bow


def train_bownn(bowextractor,
                extractor=cv2.xfeatures2d.SIFT_create()):
    '''
    该函数用于训练以BOW特征作为输入的二分类神经网络

    :param bowextractor: BOW特征提取器
    :param extractor: 特征提取器，如ORB、SIFT、SURF等
    :return:
    '''

    pos, neg = "pos-", "neg-"

    traindata, trainlabels = [], []
    print("adding to train data")
    start = time.time()
    for i in range(SAMPLES):
        bowDes_pos = bow_features(cv2.imread(path(pos, i), cv2.IMREAD_GRAYSCALE), bowextractor, extractor)
        if bowDes_pos is not None:
            traindata.extend(bowDes_pos)  # bowDes shape: (1, cluster_count)
            trainlabels.append(1)
        bowDes_neg = bow_features(cv2.imread(path(neg, i), cv2.IMREAD_GRAYSCALE), bowextractor, extractor)
        if bowDes_neg is not None:
            traindata.extend(bowDes_neg)
            trainlabels.append(-1)
    end = time.time()
    traindata = np.array(traindata)
    trainlabels = np.array(trainlabels)
    print('traindata shape:', traindata.shape)  # (799, cluster_count)
    print('trainlabels shape:', trainlabels.shape)  # (799, )
    print("训练ANN时间：", (end - start))

    return 1



