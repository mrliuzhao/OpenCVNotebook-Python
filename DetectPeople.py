import cv2
import numpy as np


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 255), thickness=5)


# img = cv2.imread(r'D:\temp\people.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread(r'.\resources\pic3.jpg', cv2.IMREAD_COLOR)
img = cv2.imread(r'.\resources\people3.jpg', cv2.IMREAD_COLOR)
hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

found, ws = hog.detectMultiScale(img)
print(len(found))
print(ws.shape)
print(ws.max())
found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
        else:
            found_filtered.append(r)

print(len(found_filtered))

for person in found_filtered:
    draw_person(img, person)

cv2.namedWindow('People Detection', cv2.WINDOW_NORMAL)
cv2.imshow('People Detection', img)


cv2.waitKey(0)
cv2.destroyAllWindows()

