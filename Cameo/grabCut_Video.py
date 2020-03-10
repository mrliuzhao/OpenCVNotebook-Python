import numpy as np
import cv2
from matplotlib import pyplot as plt


if __name__ == '__main__':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    face_xml = cv2.CascadeClassifier(r"C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_xml.detectMultiScale(gray, 1.3, 5)

        for rect in faces:
            x, y, w, h = rect
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            imgCp = frame * mask2[:, :, np.newaxis]

        cv2.imshow('display', imgCp)

        keyCode = cv2.waitKey(1)
        if keyCode == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

