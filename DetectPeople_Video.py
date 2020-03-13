import cv2
import numpy as np


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
# hog.setSVMDetector(cv2.HOGDescriptor.getDaimlerPeopleDetector())

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOCUS, 10.0)
while True:
    _, frame = cap.read()

    found, w = hog.detectMultiScale(frame)
    # 返回list of rectangle
    found_filtered = []
    # for ri, r in enumerate(found):
    #     for qi, q in enumerate(found):
    #         if ri != qi and is_inside(r, q):
    #             break
    #         else:
    #             found_filtered.append(r)
    # print(len(found_filtered))

    # for person in found_filtered:
    for person in found:
        x, y, w, h = person
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=5)

    cv2.imshow('People Detection', frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()


