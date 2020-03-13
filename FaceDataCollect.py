import cv2

# 需要使用安装opencv-python绝对路径下的xml文件
face_xml = cv2.CascadeClassifier(r'C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
faceExt_xml = cv2.CascadeClassifier(r'C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalcatface_extended.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_xml.detectMultiScale(gray, 1.3, 5)
    # facesExt = faceExt_xml.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 1:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # 将脸部区域重整为200*200的灰度图像保存
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imwrite(r'.\resources\faces\lz\%s.pgm' % str(count), f)
            count += 1

    cv2.imshow('Display', frame)

    keyCode = cv2.waitKey(1)
    if keyCode == ord('q') or keyCode == 27:
        break
    # if keyCode == ord(' '):
    #     for (x, y, w, h) in faces:
    #         # 将脸部区域重整为200*200的灰度图像保存
    #         f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    #         cv2.imwrite(r'.\resources\faces\qiqi\%s.pgm' % str(count), f)
    #         count += 1


cap.release()
cv2.destroyAllWindows()

