import cv2


def resize(img, scale_factor):
    h, w = img.shape[:2]
    h_sc = int(h/scale_factor)
    w_sc = int(w/scale_factor)
    return cv2.resize(img, (w_sc, h_sc), interpolation=cv2.INTER_AREA)


def pyramid(image, scale=1.5, min_size=(200, 80)):
    '''
    构建图片金字塔

    :param image: 图像
    :param scale: 每层缩小的比例
    :param min_size: 缩小到最小大小
    :return: 返回一个生成器，每次迭代生成按比例缩小后的图片
    '''
    yield image

    while True:
        image = resize(image, scale)
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image
