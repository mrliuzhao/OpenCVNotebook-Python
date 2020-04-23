import cv2
import numpy as np
import DigitsOCR.ocrutils as utils

try:
    l1w = np.load(r".\resources\models\mnist-digitocr-l1w.npy")
    l1b = np.load(r".\resources\models\mnist-digitocr-l1b.npy")
    # l2w = np.load(r".\resources\models\mnist-digitocr-l2w.npy")
    # l2b = np.load(r".\resources\models\mnist-digitocr-l2b.npy")
    outw = np.load(r".\resources\models\mnist-digitocr-outw.npy")
    outb = np.load(r".\resources\models\mnist-digitocr-outb.npy")
except:
    print("Cannot load model params from file")
    exit(-1)

# img_path = r'.\resources\digits.png'
# img_path = r'.\resources\numbers.jpg'
# img_path = r'.\resources\digits.jpg'
# img_path = r'.\resources\digittest.jpg'

img_path = r'.\resources\MNISTsamples.png'
# img_path = r'.\resources\numbers.gif'

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# gray = cv2.bitwise_not(gray)

ret, imgth = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
cv2.namedWindow('display', cv2.WINDOW_NORMAL)
cv2.imshow('display', imgth)

imgth2, contours, hierarchy = cv2.findContours(imgth, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
print("len of contours:", len(contours))
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    roi = imgth[y:y+h, x:x+w]
    prediction = utils.predict_digits(roi, l1w, l1b, outw, outb)

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.putText(img, str(prediction[0]), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
cv2.imshow('ori', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
