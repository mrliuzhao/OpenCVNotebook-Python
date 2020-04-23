import cv2
import struct
import numpy as np
import time
import DigitsOCR.ocrutils as utils
import tensorflow.compat.v1 as tf


imgdata, labeldata = utils.load_mnistdata('train')
testimg, testlabel = utils.load_mnistdata('test')
print("image data shape:", imgdata.shape)  # (60000, 28, 28)
print("label data shape:", labeldata.shape)  # (60000,)
print("test image data shape:", testimg.shape)  # (10000, 28, 28)
print("test label data shape:", testlabel.shape)  # (10000, )

# 取消TensorFlow2.0特性
tf.disable_v2_behavior()

# 将28*28的图片展开作为输入
imgs = tf.placeholder(dtype=tf.float32, shape=(None, 28 * 28), name='img')
labels = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='label')

layer1_nodes = 250
layer2_nodes = 50
# 第1层150个神经元
l1W = tf.Variable(tf.random_normal([28 * 28, layer1_nodes], stddev=1), dtype=tf.float32, name='layer1Weights')
l1B = tf.Variable(tf.random_normal([1, layer1_nodes], stddev=1), dtype=tf.float32, name='layer1Bias')
l1O = tf.add(tf.matmul(imgs, l1W), l1B)
l1O = tf.nn.sigmoid(l1O)

# 第2层50个神经元
# l2W = tf.Variable(tf.random_normal([layer1_nodes, layer2_nodes], stddev=1), dtype=tf.float32, name='layer2Weights')
# l2B = tf.Variable(tf.random_normal([1, layer2_nodes], stddev=1), dtype=tf.float32, name='layer2Bias')
# l2O = tf.add(tf.matmul(l1O, l2W), l2B)
# l2O = tf.nn.sigmoid(l2O)

# 输出层10个神经元，故权重为一个30 * 10的矩阵，偏置为1*10
outputW = tf.Variable(tf.random_normal([layer1_nodes, 10], stddev=1), dtype=tf.float32, name='outputWeights')
outputB = tf.Variable(tf.random_normal([1, 10], stddev=1), dtype=tf.float32, name='outputBias')
output = tf.add(tf.matmul(l1O, outputW), outputB)
# output = tf.nn.sigmoid(output)
# 使用softmax方式激活输出层，以便使用交叉熵作为损失函数
output = tf.nn.softmax(output)

# 定义损失函数
# error = tf.square(tf.subtract(labels, output))
# loss = tf.reduce_sum(error)
# 损失函数 使用交叉熵的方式
loss = tf.reduce_mean(
    -tf.reduce_sum(labels * tf.log(output), reduction_indices=1))

# 优化方法 back prop
trainstep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

# accurate  model
acc_mat = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
acc = tf.reduce_sum(tf.cast(acc_mat, tf.float32))

# 肉眼可见的测试
imgact, labelact = utils.next_batch(testimg, testlabel, batchsize=5)
# 随便展示一张图片以肉眼验证
imgtest = np.asarray(imgact[3].ravel(), dtype=np.uint8).reshape((28, -1))

imgact = tf.constant(imgact.tolist())
l1P = tf.add(tf.matmul(imgact, l1W), l1B)
l1P = tf.nn.sigmoid(l1P)
# l2P = tf.add(tf.matmul(l1P, l2W), l2B)
# l2P = tf.nn.sigmoid(l2P)
predict = tf.add(tf.matmul(l1P, outputW), outputB)
# predict = tf.nn.sigmoid(predict)
predict = tf.nn.softmax(predict)

with tf.Session() as sess:
    sess.run(init)
    start = time.time()
    for i in range(50000):
        imgbatch, labelbatch = utils.next_batch(imgdata, labeldata, batchsize=1000)
        sess.run(trainstep, feed_dict={imgs: imgbatch,
                                       labels: labelbatch})
        # 每训练1000次测试一下正确率
        if i % 1000 == 0:
            print('loss is', sess.run(loss, feed_dict={imgs: imgbatch,
                                                       labels: labelbatch}))
            imgbatch, labelbatch = utils.next_batch(testimg, testlabel, batchsize=10000)
            curr_acc = sess.run(acc, feed_dict={imgs: imgbatch,
                                                labels: labelbatch})
            print("current acc : %f %", (curr_acc / 100.0))
    end = time.time()
    print('training time: %.10f s' % (end - start))  # 21.8645482063 s

    layer1_weights = np.asarray(sess.run(l1W), dtype=np.float32)
    layer1_bias = np.asarray(sess.run(l1B), dtype=np.float32)
    # layer2_weights = np.asarray(sess.run(l2W), dtype=np.float32)
    # layer2_bias = np.asarray(sess.run(l2B), dtype=np.float32)
    output_weights = np.asarray(sess.run(outputW), dtype=np.float32)
    output_bias = np.asarray(sess.run(outputB), dtype=np.float32)
    print('shape of layer1 weights:', layer1_weights.shape)  # (784, 100)
    print('shape of layer1 bias:', layer1_bias.shape)  # (1, 100)
    # print('shape of layer2 weights:', layer2_weights.shape)  # (100, 50)
    # print('shape of layer2 bias:', layer2_bias.shape)  # (1, 50)
    print('shape of output weights:', output_weights.shape)  # (50, 10)
    print('shape of output bias:', output_bias.shape)  # (1, 10)
    np.save(r".\resources\models\mnist-digitocr-l1w.npy", layer1_weights)
    np.save(r".\resources\models\mnist-digitocr-l1b.npy", layer1_bias)
    # np.save(r".\resources\models\mnist-digitocr-l2w.npy", layer2_weights)
    # np.save(r".\resources\models\mnist-digitocr-l2b.npy", layer2_bias)
    np.save(r".\resources\models\mnist-digitocr-outw.npy", output_weights)
    np.save(r".\resources\models\mnist-digitocr-outb.npy", output_bias)

    print('Actual label:', labelact)
    print('Prediction:', sess.run(predict))

cv2.namedWindow('display', cv2.WINDOW_NORMAL)
cv2.imshow('display', imgtest)
cv2.waitKey(0)
cv2.destroyAllWindows()
