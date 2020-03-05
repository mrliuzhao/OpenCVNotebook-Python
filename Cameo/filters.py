import numpy as np
import cv2


class ImgFilter(object):
    '''filter for image'''

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        '''apply the kernel on a image'''
        cv2.filter2D(src, ddepth=-1, kernel=self._kernel, dst=dst)


class SharpenImage(ImgFilter):
    '''sharpen image'''

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], np.float32)
        ImgFilter.__init__(self, kernel=kernel)


class EmbossFilter(ImgFilter):
    '''Emboss image'''

    def __init__(self):
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]], np.float32)
        ImgFilter.__init__(self, kernel=kernel)


def strokeEdges(src, dst, blurKSize = 7, edgeKSize = 5):
    '''先进行中值滤波，再进行边缘检测  \n
    blurKSize -- 中值滤波卷积核大小  \n
    edgeKSize -- 边缘检测卷积核大小  \n
    '''
    bluredImg = cv2.medianBlur(src, blurKSize)
    gray = cv2.cvtColor(bluredImg, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(gray, ddepth=-1, dst=gray, ksize=edgeKSize)
    normalizedInverseAlpha = (255 - gray) / 255.0
    channels = cv2.split(src)
    for c in channels:
        c[:] = c * normalizedInverseAlpha
    cv2.merge(channels, dst)




