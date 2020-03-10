import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r'.\resources\cartoon1.jpg', cv2.IMREAD_COLOR)
mask = np.zeros(img.shape[:2], np.uint8)
h, w = img.shape[:2]
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

print(w, h)
# rect = (0, 0, w, h)
rect = (100, 50, w-100, h-100)
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask, bgdModel, fgdModel = cv2.grabCut(img, None, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD), 0, 1).astype('uint8')
imgCp = img*mask2[:, :, np.newaxis]

cv2.rectangle(img, (100, 50), (w-100, h-100), (0, 0, 255), 3)

cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
cv2.imshow('Segmentation', cv2.hconcat((img, imgCp)))

cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(img, cmap = "gray"),plt.colorbar(),plt.show()
