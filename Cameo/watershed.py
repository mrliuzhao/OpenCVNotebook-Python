import numpy as np
import cv2


if __name__ == '__main__':
    img = cv2.imread(r'.\resources\water_coins.jpg', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图片二值化，物体为白色
    ret, imgBi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Binarization', cv2.hconcat((img, cv2.cvtColor(imgBi, cv2.COLOR_GRAY2BGR))))

    # 开运算去除物体外的噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(imgBi, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow('Openning', cv2.hconcat((imgBi, opening)))

    # 膨胀获取确定是背景的区域，为0处
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    print('确定背景像素数目：', np.sum(sure_bg == 0))
    cv2.imshow('Sure Background', cv2.hconcat((sure_bg, cv2.bitwise_not(sure_bg))))

    # 距离变换获取确定是物体的区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist = np.uint8(255 * dist_transform / dist_transform.max())
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    cv2.imshow('Sure Foreground', cv2.hconcat((dist, sure_fg)))

    # 确定是背景和确定是物体区域外为未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imshow('background foreground unknown', cv2.hconcat((sure_bg, sure_fg, unknown)))

    # 对确定是物体的区域进行不同数字的标记
    ret, markers = cv2.connectedComponents(sure_fg)
    print('总共物体区域数目：', markers.max())
    # 不同区域物体使用不同颜色表示出来
    colors = np.int32(list(np.ndindex(2, 2, 2))) * 255
    # 构建调色板，第一个颜色为黑色，第二个为白色
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.int32)
    for i in range(50):
        palette = np.concatenate([palette, colors[1:len(colors)-1]], axis=0)

    # 每个区域ID+1，使得背景区域标记为1
    markers = markers + 1
    # 再把未知的区域标记回0
    markers[unknown == 255] = 0
    markers_col = np.uint8(palette[markers])
    cv2.imshow('marker', markers_col)

    # 进行分水岭算法
    cv2.watershed(img, markers)
    print('分水岭计算后，最大值：{}；最小值：{}'.format(markers.max(), markers.min()))

    # 边缘为-1，加一处理，S结果中边缘为黑色
    markers = np.where(markers == -1, 0, markers)
    markers_col = np.uint8(palette[markers])
    cv2.imshow('water shed', markers_col)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

# class App:
#     def __init__(self, fn):
#         self.img = cv.imread(fn)
#         if self.img is None:
#             raise Exception('Failed to load image file: %s' % fn)
#
#         h, w = self.img.shape[:2]
#         self.markers = np.zeros((h, w), np.int32)
#         self.markers_vis = self.img.copy()
#         self.cur_marker = 1
#         self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255
#
#         self.auto_update = True
#         self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors)
#
#     def get_colors(self):
#         return list(map(int, self.colors[self.cur_marker])), self.cur_marker
#
#     def watershed(self):
#         m = self.markers.copy()
#         cv.watershed(self.img, m)
#         overlay = self.colors[np.maximum(m, 0)]
#         vis = cv.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv.CV_8UC3)
#         cv.imshow('watershed', vis)
#
#     def run(self):
#         while cv.getWindowProperty('img', 0) != -1 or cv.getWindowProperty('watershed', 0) != -1:
#             ch = cv.waitKey(50)
#             if ch == 27:
#                 break
#             if ch >= ord('1') and ch <= ord('7'):
#                 self.cur_marker = ch - ord('0')
#                 print('marker: ', self.cur_marker)
#             if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
#                 self.watershed()
#                 self.sketch.dirty = False
#             if ch in [ord('a'), ord('A')]:
#                 self.auto_update = not self.auto_update
#                 print('auto_update if', ['off', 'on'][self.auto_update])
#             if ch in [ord('r'), ord('R')]:
#                 self.markers[:] = 0
#                 self.markers_vis[:] = self.img
#                 self.sketch.show()
#         cv.destroyAllWindows()


