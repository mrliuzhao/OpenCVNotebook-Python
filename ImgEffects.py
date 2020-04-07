import cv2

cap = cv2.VideoCapture(0)

face_xml = cv2.CascadeClassifier(
    r'C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_xml.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]
        # 马赛克效果
        mosaic_size = 10
        for i in range(0, h, mosaic_size):
            for j in range(0, w, mosaic_size):
                roi[i:min(i + mosaic_size - 1, h), j:min(j + mosaic_size - 1, w)] = roi[i, j]
        frame[y:y + h, x:x + w] = roi

    cv2.imshow('Display', frame)

    keyCode = cv2.waitKey(1)
    if keyCode == 27:
        break

cap.release()
cv2.destroyAllWindows()
