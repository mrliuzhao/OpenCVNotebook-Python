
def sliding_window(image, stepSize, windowSize):
    '''
    在图片中生成滑动窗口

    :param image: 图像数据
    :param stepSize: 每次窗口移位距离，x和y方向上每次移位距离一致
    :param windowSize: 每次生成的窗口大小
    :return: 返回生成器，每次迭代返回滑动窗口的位置：（x1,y1,roi）分别为窗口左上角的点坐标以及窗口内具体的图像数据
    '''

    h, w = image.shape[:2]
    for y in range(0, h, stepSize):
        for x in range(0, w, stepSize):
            yield (x, y, image[y:min(y + windowSize[1], h), x:min(x + windowSize[0], w)])
