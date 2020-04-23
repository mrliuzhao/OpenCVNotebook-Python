import numpy as np


def sigmoid(mat):
    '''
    对一个矩阵中所有的元素进行sigmoid函数计算，为防止溢出做了优化

    :param mat: 矩阵
    :return: 输出每个元素sigmoid的矩阵
    '''
    matCp = mat.copy()
    h, w = matCp.shape
    for i in range(h):
        for j in range(w):
            if mat[i, j] >= 0:
                matCp[i, j] = 1.0 / (1 + np.exp(-mat[i, j]))
            else:
                matCp[i, j] = np.exp(mat[i, j]) / (1 + np.exp(mat[i, j]))
    return matCp


def convolution(mat, kernel):
    '''
    对二维矩阵进行Valid模式卷积操作，卷积核大小应为奇数(2a+1)，(h, w)大小的矩阵卷积后为(h-2a, w-2a)

    :param mat: 要进行卷积的二维矩阵
    :param kernel: 卷积操作核，大小应为奇数
    :return: 返回卷积后的二维矩阵
    '''

    h, w = mat.shape[:2]
    kh, kw = kernel.shape[:2]
    if kh % 2 != 1 or kw % 2 != 1:
        print("卷积核大小应为奇数")
        return None
    conv_mat = []
    for i in range(h - kh + 1):
        row_temp = []
        for j in range(w - kw + 1):
            row_temp.append(np.sum(mat[i:i+kh, j:j+kw] * kernel))
        conv_mat.append(row_temp)
    return np.asarray(conv_mat, dtype=np.float32)


def maxpooling(mat, kernel_size=2):
    '''
    对二维矩阵进行最大值池化操作，返回最大值掩模，小于0表示该位置不为最大值；
    大于等于0时表示该位置为最大值，且对应着池化后矩阵中元素下标（1维下标）

    :param mat: 矩阵
    :param kernel_size: 池化核心大小，应为矩阵宽高的公因子
    :return: 返回池化后的矩阵，以及选择最大值的掩模
    '''

    h, w = mat.shape[:2]
    if h % kernel_size != 0 or w % kernel_size != 0:
        print("池化核心大小，应为矩阵宽高的公因子")
        return None

    pool_mat = []
    mask = np.zeros_like(mat)
    count = 0
    for i in range(0, h, kernel_size):
        row_temp = []
        for j in range(0, w, kernel_size):
            row_temp.append(np.max(mat[i:i+kernel_size, j:j+kernel_size]))
            idx = np.argmax(mat[i:i + kernel_size, j:j + kernel_size])
            r = idx // kernel_size
            c = idx % kernel_size
            mask[i+r, j+c] = count + 1
            count += 1
        pool_mat.append(row_temp)

    return np.asarray(pool_mat, dtype=np.float32), mask - 1


def train(img, label, conv_kernel, conv_bias, pool_size, outW, outB):
    '''
    使用一个卷积层、一个池化层、一个输出层作为模型训练参数

    :param img: 训练图像
    :param label: 真实标记
    :param conv_kernel: 卷积层核心
    :param conv_bias: 卷积层偏置
    :param pool_size: 最大值池化层核心大小
    :param outW: 输出层权重
    :param outB: 输出层偏置
    :return: 返回平方均值误差，以及更新后的各层参数
    '''

    img_conv = convolution(img, conv_kernel) + conv_bias
    img_act = sigmoid(img_conv)

    img_pool, pool_mask = maxpooling(img_act, pool_size)
    img_pool = img_pool.reshape((1, -1))

    outval = np.matmul(img_pool, outW) + outB
    outval = sigmoid(outval)

    # 使用平方差均值作为误差
    mse = 0.5 * np.sum(np.square(outval - label))

    # 从输出层开始逆向求梯度 - 误差反向传播
    grad_outB = (outval - label) * outval * (1 - outval)
    grad_outW = np.matmul(img_pool.T, grad_outB)

    # 遍历生成卷积层梯度
    r, c = pool_mask.shape
    grad_conv = np.zeros_like(pool_mask, dtype=np.float32)
    for i in range(r):
        for j in range(c):
            if pool_mask[i, j] >= 0:
                temp = grad_outB * outW[int(pool_mask[i, j]), :]
                grad_conv[i, j] = np.sum(temp) * img_act[i, j] * (1 - img_act[i, j])
    grad_convB = np.sum(grad_conv)

    # 遍历生成卷积核的梯度
    r, c = conv_kernel.shape
    h, w = img_act.shape
    grad_convW = np.zeros_like(conv_kernel, dtype=np.float32)
    for i in range(r):
        for j in range(c):
            temp = img[i:i + h, j:j + w] * grad_conv
            grad_convW[i, j] = np.sum(temp)

    # 根据梯度更新参数
    learning_rate = 0.1
    conv_kernel = conv_kernel - (learning_rate * grad_convW)
    conv_bias = conv_bias - (learning_rate * grad_convB)

    outW = outW - (learning_rate * grad_outW)
    outB = outB - (learning_rate * grad_outB)

    return mse, conv_kernel, conv_bias, outW, outB


def predict(img, label, conv_kernel, conv_bias, pool_size, outW, outB):
    '''
    使用训练好的模型预测图片

    :param img: 训练图像
    :param label: 真实标记
    :param conv_kernel: 卷积层核心
    :param conv_bias: 卷积层偏置
    :param pool_size: 最大值池化层核心大小
    :param outW: 输出层权重
    :param outB: 输出层偏置
    :return: 返回平方均值误差，以及输出值
    '''

    img_conv = convolution(img, conv_kernel) + conv_bias
    img_act = sigmoid(img_conv)

    img_pool, pool_mask = maxpooling(img_act, pool_size)
    img_pool = img_pool.reshape((1, -1))

    outval = np.matmul(img_pool, outW) + outB
    outval = sigmoid(outval)

    # 使用平方差均值作为误差
    mse = 0.5 * np.sum(np.square(outval - label))

    return mse, outval

