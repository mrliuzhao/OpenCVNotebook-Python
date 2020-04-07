import cv2
import numpy as np
from os.path import join
import CarDetection.pyramid as py


img = cv2.imread(r".\resources\car1.jpg", cv2.IMREAD_COLOR)
h, w = img.shape[:2]
cv2.imshow('display', img)

count = 0
for image in py.pyramid(img, 1.5, (w/5, h/5)):
    fn = 'car-%d.jpg' % count
    cv2.imwrite(fn, image)
    count += 1



cv2.waitKey(0)
cv2.destroyAllWindows()




