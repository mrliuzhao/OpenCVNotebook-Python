import cv2

cap = cv2.VideoCapture(0)

mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)

morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    _, frame = cap.read()

    # fgmask = mog.apply(frame)
    fgmask = knn.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, morph_kernel)
    fgmask, contours, hierarchy = cv2.findContours(fgmask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 40 * 40:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 0, 255), 2)

    extract = cv2.bitwise_and(frame, frame, mask=fgmask)

    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Display', cv2.hconcat((frame, fgmask, extract)))

    keyCode = cv2.waitKey(1)
    if keyCode == 27:
        break

cap.release()
cv2.destroyAllWindows()









