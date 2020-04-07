import cv2

cap = cv2.VideoCapture(0)

background = None
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    _, frame = cap.read()

    if background is None:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, (21, 21), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 1)

    diff = cv2.absdiff(gray, background)
    _, diff = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, morph_kernel)

    diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Display', cv2.hconcat((frame, diff)))

    keyCode = cv2.waitKey(1)
    if keyCode == 27:
        break

cap.release()
cv2.destroyAllWindows()









