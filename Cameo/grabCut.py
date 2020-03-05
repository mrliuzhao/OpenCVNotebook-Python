import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r'.\resources\cartoon1.jpg', cv2.IMREAD_COLOR)
mask = np.zeros(img.shape[:2], np.uint8)
h, w = img.shape[:2]
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# rect = (0, 0, w, h)
rect = (100, 50, 421, 378)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
imgCp = img*mask2[:, :, np.newaxis]

cv2.imshow('Segmentation', cv2.hconcat((img, imgCp)))

cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow(img, cmap = "gray"),plt.colorbar(),plt.show()
