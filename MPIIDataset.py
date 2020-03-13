import cv2
import numpy as np
import scipy.io as scio


data = scio.loadmat(r'.\resources\mpii_human_pose_v1_u12_1.mat')
dataRel = data['RELEASE']
annolist = dataRel['annolist'][0, 0]
video_list = dataRel['video_list'][0, 0]
img_train = dataRel['img_train'][0, 0]
version = dataRel['version'][0, 0]
single_person = dataRel['single_person'][0, 0]
act = dataRel['act'][0, 0]

print('len of annoList:', len(annolist.ravel()))
print('shape of annoList:', annolist.shape)
for i in range(len(annolist.ravel())):
    anno = annolist[0, i]
    annRect = anno['annorect']
    rows, cols = annRect.shape
    if rows > 1 or cols > 1:
        print('annRect.shape:', annRect.shape)  # (1, 2)
        print('annRect.dtype:', annRect.dtype)  # [('scale', 'O'), ('objpos', 'O')]
        print('annRect ravel:', annRect.ravel())
        print('第{}个annRect维度：{},{}'.format(i, rows, cols))

        imgName = anno['image']['name'][0, 0][0]
        path = r'F:\Downloads\images\%s' % imgName
        print('Going to get image:', path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        for j in range(cols):
            humanPosX = annRect[0, j]['objpos']['x'][0, 0][0, 0]
            humanPosY = annRect[0, j]['objpos']['y'][0, 0][0, 0]
            print(humanPosX)
            print(humanPosY)
            cv2.circle(img, (humanPosX, humanPosY), 8, (0, 0, 255), cv2.FILLED)

        # for x, y in zip(humanPosX, humanPosY):
        #     print(x, y)
        #

        cv2.imshow('Display', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        break


# imgIdx = 100
# first = annolist[0, imgIdx]
# firstImgName = first['image']['name'][0, 0][0]
#
# path = r'F:\Downloads\images\%s' % firstImgName
# print('Going to get image:', path)
#
# img = cv2.imread(path, cv2.IMREAD_COLOR)
#
# firstAnno = first['annorect']
# humanPosX = firstAnno['objpos'][0, 0]['x'][0, 0][0]
# humanPosY = firstAnno['objpos'][0, 0]['y'][0, 0][0]
# print(humanPosX)
# print(humanPosY)
#
# for x, y in zip(humanPosX, humanPosY):
#     print(x, y)
#     cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
#
#
# cv2.imshow('Display', img)
#
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()







