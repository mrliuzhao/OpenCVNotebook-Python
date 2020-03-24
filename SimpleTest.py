import cv2
import numpy as np
from os.path import join


datapath = r".\resources\CarData\TrainImages"


def path(cls, i):
    return "%s\\%s-%d.pgm" % (datapath, cls, i)


pos, neg = "pos", "neg"
# SIFT特征
sift = cv2.xfeatures2d.SIFT_create()


print("path:", path(pos, 0))
img = cv2.imread(path(pos, 0), cv2.IMREAD_GRAYSCALE)
print(img.shape)
img = cv2.imread(path(pos, 1), cv2.IMREAD_GRAYSCALE)
print(img.shape)
kpts, des = sift.detectAndCompute(img, mask=None)
print("length of des:", len(des))





