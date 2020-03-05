from Cameo.filters import EmbossFilter
import cv2
from matplotlib import pyplot as plt
import os


if __name__ == "__main__":
    path = os.path.abspath('..')
    path = os.path.join(path, 'test', 'abc.jpg')
    print(path)

    img = cv2.imread(r".\resources\pic2.jpg", cv2.IMREAD_COLOR)
    # img = cv2.imread(r".\resources\left-1.jpg")
    # cv2.imshow('Left', img)
    # img = cv2.imread(r".\resources\heart.png", cv2.IMREAD_COLOR)
    hrt = cv2.imread(r".\resources\heart.png", cv2.IMREAD_UNCHANGED)
    hh, hw = hrt.shape[:2]
    b, g, r, a = cv2.split(hrt)

    hrt = cv2.merge((b, g, r))
    # alphaInv = cv2.bitwise_not(a)
    # cv2.imshow('AlphaInverse', alphaInv)
    hrt = cv2.bitwise_and(hrt, hrt, mask=a)
    b, g, r = cv2.split(hrt)
    # ret, mask = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)
    alphaInv = cv2.bitwise_not(r)
    cv2.imshow('AlphaInverse', alphaInv)

    eye_xml = cv2.CascadeClassifier(r"C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_eye.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_xml.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in eyes:
        scale = min(w / hw, h / hh)
        print('x, y, w, h: {}, {}, {}, {}'.format(x, y, w, h))
        print('scale:', scale)
        if scale > 1:
            scHrt = cv2.resize(hrt, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            scMask = cv2.resize(alphaInv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            scHrt = cv2.resize(hrt, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            scMask = cv2.resize(alphaInv, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow('reshaped heart', scHrt)
        cv2.imshow('reshaped alpha', scMask)

        xEnd = int(x + hw * scale)
        yEnd = int(y + hh * scale)
        roi = img[y:yEnd, x:xEnd, :]
        print('scHrt shape:', scHrt.shape)
        print('scMask shape:', scMask.shape)
        print('roi shape:', roi.shape)
        cv2.imshow('ROI Before', roi)
        roi = cv2.bitwise_and(roi, roi, mask=scMask)
        cv2.imshow('ROI After', roi)
        sub = cv2.add(roi, scHrt)
        img[y:yEnd, x:xEnd, :] = sub

    cv2.imshow('Eye Detect', img)



    cv2.waitKey(0)
    cv2.destroyAllWindows()

