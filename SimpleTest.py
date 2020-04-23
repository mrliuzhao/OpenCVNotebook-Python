import cv2
import numpy as np
import time
import tensorflow as tf
import DigitsOCR.MatUtils as matutils
import DigitsOCR.ocrutils as utils

# conv_kernel = np.random.normal(0, 1, size=(6, 6))
# maxpool, mask = matutils.maxpooling(conv_kernel, 3)
# print(conv_kernel)
# print(maxpool)
# print(mask)

testimg, testlabel = utils.load_mnistdata('test')
imgbatch, labelbatch = utils.next_batch(testimg, testlabel, batchsize=100)

print('test image shape:', imgbatch.shape)
print('test label shape:', labelbatch.shape)




