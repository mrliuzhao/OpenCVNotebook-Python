import cv2

# 需要使用安装opencv-python绝对路径下的xml文件
face_xml = cv2.CascadeClassifier(r'C:\Program Files\Python38\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier(r"C:\Program Files\Python38\Lib\site-packages\cv2\data\haarcascade_eye.xml")

img = cv2.imread(r'D:/temp/people2.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread(r'.\resources\pic3.jpg', cv2.IMREAD_COLOR)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_xml.detectMultiScale(imgGray, 1.3, 5)
eyes = eye_xml.detectMultiScale(imgGray, 1.3, 5)

print(len(faces))
print(len(eyes))
for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)

for (x, y, w, h) in eyes:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 255), thickness=2)

cv2.imshow('Face Detection', img)


cv2.waitKey(0)
cv2.destroyAllWindows()
