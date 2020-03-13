import cv2
import os
import numpy as np


def load_img(path):
    labelNameDict = {}
    labels = []
    faceImgs = []
    count = 0
    for root, dirs, files in os.walk(path):
        if root == path:
            continue

        name = root.split('\\')[-1]
        # print(name)
        labelNameDict[count] = name
        for f in files:
            grayImg = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
            # faceDict[name].append(os.path.join(root, f))
            faceImgs.append(grayImg)
            labels.append(count)
        count += 1

    return faceImgs, labels, labelNameDict


if __name__ == "__main__":

    faceImgs, labels, labelNameDict = load_img(r'.\resources\faces')

    faceImgs = np.asarray(faceImgs)
    labels = np.asarray(labels)

    # eigenFaceModel
    # model = cv2.face.EigenFaceRecognizer_create()
    # FisherFaceModel
    # model = cv2.face.FisherFaceRecognizer_create()
    # LBPHFaceRecognizer
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faceImgs, labels)

    # 使用另一半数据检测，并给出阈值范围
    face_test = faceImgs[len(faceImgs)//2:]
    label_test = labels[len(labels)//2:]
    errorCount = 0
    confList = []
    for face, lab in zip(face_test, label_test):
        label, confidence = model.predict(face)
        if label != lab:
            # print('Error detection!!!')
            errorCount += 1
        else:
            confList.append(confidence)

    print('Total count:', len(face_test))
    print('Total count2:', len(label_test))
    print('Error count:', errorCount)
    print('minimal confidence:', min(confList))
    print('maximal confidence:', max(confList))
    print('mean of confidence:', np.mean(confList))
    print('median of confidence:', np.median(confList))
    print('variance of confidence:', np.var(confList))

    # path = r'.\resources\faces\lz'
    # for root, dirs, files in os.walk(path):
    #     for f in files:
    #         grayImg = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
    #         label, confidence = model.predict(grayImg)
    #         print('Predict label:{}; confidence:{}'.format(label, confidence))


    # cap = cv2.VideoCapture(0)
    # face_xml = cv2.CascadeClassifier(
    #     r'C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    #
    # while True:
    #     ret, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_xml.detectMultiScale(gray, 1.3, 5)
    #     for (x, y, w, h) in faces:
    #         # EigenFace要求预测数据大小与训练集图片大小一致
    #         roi = gray[y:y + h, x:x + w]
    #         # roi = cv2.resize(roi, (200, 200))
    #         label, confidence = model.predict(roi)
    #         print('Predict label:{}; confidence:{}'.format(label, confidence))
    #         # labelNameDict[label]
    #         if confidence <= 50.0:
    #             cv2.putText(frame, labelNameDict[label], (x-5, y-5), fontFace=cv2.FONT_HERSHEY_COMPLEX,
    #                         fontScale=1, thickness=2, color=(0, 0, 255))
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #
    #     cv2.imshow('Display', frame)
    #
    #     keyCode = cv2.waitKey(1)
    #     if keyCode == 27:
    #         break
    #
    #
    # cap.release()
    # cv2.destroyAllWindows()



