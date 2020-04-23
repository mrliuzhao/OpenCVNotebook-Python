import struct
import numpy as np
import random
import tensorflow.compat.v1 as tf
import cv2


def load_mnistdata(type='train'):
    '''
    从MINST源数据文件中解析数据

    :param type: 数据类型，train时解析训练数据，test时解析检测数据
    :return: 第一个值为解析后的图像数据，第二个值为解析后的标签数据
    '''

    if type == 'train':
        origin_img_file_path = r'D:\PythonWorkspace\OpenCVNotebook-Python\resources\MNISTData\train-images.idx3-ubyte'
        origin_label_file_path = r'D:\PythonWorkspace\OpenCVNotebook-Python\resources\MNISTData\train-labels.idx1-ubyte'
        img_npy_path = r'D:\PythonWorkspace\OpenCVNotebook-Python\resources\mnist-img-train.npy'
        label_npy_path = r'D:\PythonWorkspace\OpenCVNotebook-Python\resources\mnist-label-train.npy'
    elif type == 'test':
        origin_img_file_path = r'D:\PythonWorkspace\OpenCVNotebook-Python\resources\MNISTData\t10k-images.idx3-ubyte'
        origin_label_file_path = r'D:\PythonWorkspace\OpenCVNotebook-Python\resources\MNISTData\t10k-labels.idx1-ubyte'
        img_npy_path = r'D:\PythonWorkspace\OpenCVNotebook-Python\resources\mnist-img-test.npy'
        label_npy_path = r'D:\PythonWorkspace\OpenCVNotebook-Python\resources\mnist-label-test.npy'
    else:
        print("Error type! type should only be train or test")
        return None

    imgdata = None
    labels = None
    try:
        imgdata = np.load(img_npy_path)
        labels = np.load(label_npy_path)
    except:
        print("Cannot load data from file")

    if imgdata is not None and labels is not None:
        return imgdata, labels

    # 若无法从文件读取解析后的数据矩阵，则重新从源文件解析
    if imgdata is None or labels is None:
        # 以二进制格式全部读取文件
        img_file = open(origin_img_file_path, 'rb')
        buff_img = img_file.read()
        img_file.close()

        label_file = open(origin_label_file_path, 'rb')
        buff_label = label_file.read()
        label_file.close()

        offset_img = 0
        magic_num, image_num, rown, coln = struct.unpack_from('>IIII', buff_img, offset_img)
        offset_img += struct.calcsize('>IIII')

        offset_label = 0
        magic_num_label, label_num = struct.unpack_from('>II', buff_label, offset_label)
        offset_label += struct.calcsize('>II')

        if image_num != label_num:
            print("Data is not corrected loaded! Images Number not equals to label number!!!")
            exit(-1)

        # 按行解析图片
        labels = []
        imgdata = []
        for i in range(image_num):
            img_mat = []
            for j in range(coln):
                new_row = []
                for k in range(rown):
                    new_row.append(struct.unpack_from('>B', buff_img, offset_img)[0])
                    offset_img += struct.calcsize('>B')
                img_mat.append(new_row)
            imgdata.append(img_mat)
            labels.append(struct.unpack_from('>B', buff_label, offset_label)[0])
            offset_label += struct.calcsize('>B')
        imgdata = np.asarray(imgdata, dtype=np.uint8)
        labels = np.asarray(labels, dtype=np.uint8)
        np.save(img_npy_path, imgdata)
        np.save(label_npy_path, labels)

    return imgdata, labels


def next_batch(imgdata, labeldata, batchsize=10):
    '''
    从给定的图像和标签数据中随机获取一个batch的数据并返回

    :param imgdata:  图像数据，大小应为（imgCount, imgRows, imgCols）
    :param labeldata:  标签数据，大小应为（imgCount,）
    :param batchsize:  获取数据量
    :return: 返回符合数据量大小的数据，并将图像数据展开，返回大小为（batchsize, imgRows*imgCols）;
    返回的标签数据为（batchsize, 10）
    '''

    totlen = len(imgdata)
    if batchsize < totlen:
        idx = random.sample(range(totlen), batchsize)
    else:
        idx = [random.randint(0, totlen-1) for _ in range(batchsize)]

    imgbatch = []
    labelbatch = []
    for i in idx:
        imgbatch.append(imgdata[i].ravel())
        labeltemp = np.zeros((10, ))
        labeltemp[labeldata[i]] = 1.0
        labelbatch.append(labeltemp)

    imgbatch = np.asarray(imgbatch, dtype=np.float32)
    labelbatch = np.asarray(labelbatch, dtype=np.float32)
    return imgbatch, labelbatch


def extract_digit(roi, padding=0.05):
    '''
    修正包含数字的区域为正方形，将数字置中，并在周围添加指定大小的padding

    :param roi: 包含数字的感兴趣区域
    :param padding: 边界大小，百分比
    :return: 返回数字置中的正方形图像
    '''
    h, w = roi.shape[:2]
    newlen = int(max(w, h) * (1+padding))
    img = np.zeros((newlen, newlen))
    x = int((newlen - w)/2)
    y = int((newlen - h)/2)
    img[y:y+h, x:x+w] = roi
    return img


def predict_digits(roi, l1W, l1B, outW, outB):
    digit_img = extract_digit(roi)
    if roi.shape[0] > 28:
        digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_LINEAR)

    # 取消TensorFlow2.0特性
    tf.disable_v2_behavior()

    # 随便展示一张图片以肉眼验证
    imgtest = np.asarray(digit_img.ravel(), dtype=np.float32).reshape((1, -1))
    l1P = tf.add(tf.matmul(imgtest, l1W), l1B)
    l1P = tf.nn.sigmoid(l1P)
    output = tf.add(tf.matmul(l1P, outW), outB)
    output = tf.nn.softmax(output)
    prediction = tf.argmax(output, 1)

    with tf.Session() as sess:
        res = sess.run(prediction)

    return res
