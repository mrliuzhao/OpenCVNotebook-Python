import numpy as np
import cv2


def copyRect(src, dst, srcRect=None, dstRect=None, mask=None, lockRatio=False):
    '''
    将源图（src）中的某个矩形区域拷贝到目标图（dst）中的某个矩形区域
    :param src: 要拷贝的源图
    :param dst: 拷贝到的目标图
    :param srcRect: 要拷贝源图中的矩形区域，格式为(x, y, w, h)，None时表示将源图全部拷贝
    :param dstRect: 拷贝到目标图中的矩形区域，格式为(x, y, w, h)，None时表示将目标图占满
    :param mask: 掩模，仅将源图区域内不为0的部分拷贝至目标图，为None时则直接覆盖
    :param lockRatio: 是否锁定源图拷贝区域的长宽比，保证长宽比时则不一定完全铺满目标图区域
    :return: 没有返回值，在目标图上直接修改
    '''

    if srcRect is not None:
        x, y, w, h = srcRect
        src = src[y:y+h, x:x+w]
    if dstRect is None:
        h, w = dst.shape[:2]
        dstRect = (0, 0, w, h)
    sh, sw = src.shape[:2]
    dx, dy, dw, dh = dstRect

    if mask is None:
        mask = np.ones((sh, sw), dtype=np.float32)

    # 锁定源图长宽比时，x、y方向缩放比例一致
    if lockRatio:
        # 取最小缩放比例，并重新计算目标区域大小
        scale = min(dw / sw, dh / sh)
        dw = int(round(sw * scale))
        dh = int(round(sh * scale))
        # 按照OpenCV官方说法，插值方法在放大时选择INTER_CUBIC，缩小时选择INTER_AREA
        if scale > 1:
            src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        src = cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_LINEAR)

    dstCp = dst[dy:dy+dh, dx:dx+dw]
    # # 将目标图中将要被覆盖的部分抠掉
    # dstCp = cv2.bitwise_and(dstCp, dstCp, mask=cv2.bitwise_not(mask))
    # # 再将源图要拷贝部分添加进来
    # dstCp = cv2.add(dstCp, src)
    # dst[dy:dh, dx:dw] = dstCp
    # 使用numpy.where实现类似操作
    # 如果是彩色图像则扩展mask为3通道，以便使用numpy.where
    if len(src.shape) == 3 and src.shape[2] == 3:
        mask = mask.repeat(3).reshape(dh, dw, 3)
    dst[dy:dy+dh, dx:dx+dw] = np.where(mask, src, dstCp)





def createMedianMask(disparityMap, validDepthMask, rect = None):
    '''
    该函数用于生成Mask，忽略指定区域内距离过远或过近，或深度数据无效的像素
    :param disparityMap: 视差数据
    :param validDepthMask: 深度有效性数据，0表示深度无效，其他值表示深度有效
    :param rect: 关心的区域，格式为(x,y,w,h)，None表示全部关心
    :return:
    '''
    # 取出ROI
    if rect is not None:
        x, y, w, h = rect
        disparityMap = disparityMap[y:y+h, x:x+w]
        validDepthMask = validDepthMask[y:y+h, x:x+w]
    median = np.median(disparityMap)
    # 判断条件：深度数据无效 或 视差值与中值之差过大或过小（距离过近或过远）时均为0，即忽略
    return np.where((validDepthMask == 0) |
                    (abs(disparityMap - median) >= 12),
                    0.0, 1.0)









