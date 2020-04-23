import cv2
import numpy as np
import time
import DigitsOCR.ocrutils as utils
import DigitsOCR.MatUtils as matutils

imgdata, labeldata = utils.load_mnistdata('train')
testimg, testlabel = utils.load_mnistdata('test')
print("image data shape:", imgdata.shape)  # (60000, 28, 28)
print("label data shape:", labeldata.shape)  # (60000,)
print("test image data shape:", testimg.shape)  # (10000, 28, 28)
print("test label data shape:", testlabel.shape)  # (10000, )

# 模型为一个卷积层，卷积核5*5
conv_kernel = np.random.normal(0, 1, size=(5, 5))
conv_bias = np.random.normal(0, 1, 1)[0]
# 3*3，最大值池化
# 输出层10个神经元，以池化层作为输入，输出分类one-hot
outW = np.random.normal(0, 1, size=(64, 10))
outB = np.random.normal(0, 1, size=(1, 10))

for i in range(20000):
    img = np.asarray(imgdata[i].ravel(), dtype=np.float32).reshape((28, -1))
    label = np.asarray(labeldata[i].ravel(), dtype=np.float32).reshape((1, -1))
    mse, conv_kernel, conv_bias, outW, outB = matutils.train(img, label, conv_kernel, conv_bias, 3, outW, outB)
    if i % 100 == 0:
        print('训练次数：%d; 误差：%f' % (i+1, mse))


imgbatch, labelbatch = utils.next_batch(testimg, testlabel, batchsize=100)
correct_count = 0
for i in range(100):
    imgtest = np.asarray(imgbatch[i].ravel(), dtype=np.uint8).reshape((28, -1))
    labeltest = np.asarray(labelbatch[i].ravel(), dtype=np.float32).reshape((1, -1))
    mse, outval = matutils.predict(imgtest, labeltest, conv_kernel, conv_bias, 3, outW, outB)
    if np.argmax(outval) == np.argmax(labeltest):
        correct_count += 1
print('正确率：%f' % (correct_count / 100))

# 随便展示一张图片以肉眼验证
imgtest = np.asarray(imgbatch[3].ravel(), dtype=np.uint8).reshape((28, -1))
labeltest = np.asarray(labelbatch[3].ravel(), dtype=np.float32).reshape((1, -1))

mse, outval = matutils.predict(imgtest, labeltest, conv_kernel, conv_bias, 3, outW, outB)
print('真实值标签：', labeltest)
print('预测值：', outval)
print('预测误差：', mse)

cv2.namedWindow('display', cv2.WINDOW_NORMAL)
cv2.imshow('display', imgtest)
cv2.waitKey(0)
cv2.destroyAllWindows()


