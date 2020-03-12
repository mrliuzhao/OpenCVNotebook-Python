import cv2
import matplotlib.pyplot as plt


def doNothing(value):
    pass


# img = cv2.imread(r'.\resources\right-3.jpg', cv2.IMREAD_COLOR)
img = cv2.imread(r'.\resources\card.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread(r'.\resources\sudoku.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
cv2.createTrackbar('blockSize', 'Display', 2, 20, doNothing)
cv2.createTrackbar('kSize', 'Display', 1, 15, doNothing)
cv2.createTrackbar('k', 'Display', 4, 200, doNothing)

while True:

    blockSize = int(cv2.getTrackbarPos('blockSize', 'Display'))
    kSize = int(cv2.getTrackbarPos('kSize', 'Display'))
    kSize = 2 * kSize + 1
    k = cv2.getTrackbarPos('k', 'Display') / 100.0

    dst = cv2.cornerHarris(gray, blockSize, kSize, k)
    imgCp1 = img.copy()
    imgCp1[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('Display', imgCp1)
    if cv2.waitKey(1) == 27:
        break

# ksize越小越敏感 3-31之间的奇数
# dst = cv2.cornerHarris(gray, 2, 23, 0.04)
# imgCp1 = img.copy()
# imgCp1[dst > 0.01*dst.max()] = [0, 0, 255]
#
# dst2 = cv2.cornerHarris(gray, 2, 23, 0.04)
# imgCp2 = img.copy()
# imgCp2[dst2 > 0.01*dst2.max()] = [0, 0, 255]
#
# plt.hist(dst.ravel())
# plt.show()
#
# cv2.imshow('Display', img)

cv2.destroyAllWindows()


