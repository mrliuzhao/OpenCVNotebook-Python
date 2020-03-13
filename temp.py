import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

# svmDet = cv2.HOGDescriptor.getDefaultPeopleDetector()
# print('type of SVM:', type(svmDet))  # numpy.ndarray
# print('shape of SVM:', svmDet.shape)  # (3781, 1)
# print('dtype of SVM:', svmDet.dtype)  # float32
# print('SVM[0,0]:', svmDet[0, 0])
# print('max SVM:', svmDet.max())
# print('min SVM:', svmDet.min())
# plt.hist(svmDet.ravel())
# plt.show()

data = scio.loadmat(r'.\resources\mpii_human_pose_v1_u12_1.mat')
print(type(data))  # <class 'dict'>
print(type(data['RELEASE']))  # <class 'numpy.ndarray'>
print(data['RELEASE'].shape)  # (1, 1)
print(data['RELEASE'].dtype)  # [('annolist', 'O'), ('img_train', 'O'), ('version', 'O'), ('single_person', 'O'), ('act', 'O'), ('video_list', 'O')]

dataRel = data['RELEASE']
annolist = dataRel['annolist'][0, 0]
video_list = dataRel['video_list'][0, 0]
img_train = dataRel['img_train'][0, 0]
version = dataRel['version'][0, 0]
single_person = dataRel['single_person'][0, 0]
act = dataRel['act'][0, 0]

print('type of single_person:', type(single_person))
print('shape of single_person:', single_person.shape)  # (24987, 1)
print('dtype of single_person:', single_person.dtype)  # [('name', 'O')]


first = single_person[0, 0]
print('type of first single_person:', type(first))
print('shape of first single_person:', first.shape)  # (1, 1)
print('dtype of first single_person:', first.dtype)  # uint8
print('first single_person:', first[0, 0])  # uint8


print('type of annolist:', type(annolist))
print('shape of annolist:', annolist.shape)  # (1, 24987)
print('dtype of annolist:', annolist.dtype)  # [('image', 'O'), ('annorect', 'O'), ('frame_sec', 'O'), ('vididx', 'O')]

first = annolist[0, 0]
print('type of first image:', type(first['image']))
print('shape of first image:', first['image'].shape)  # (1, 1)
print('dtype of first image:', first['image'].dtype)  # [('name', 'O')]

firstImg = first['image']
print('type of first image name:', type(firstImg['name']))
print('shape of first image name:', firstImg['name'].shape)  # (1, 1)
print('dtype of first image name:', firstImg['name'].dtype)  # object
print('first image name:', firstImg['name'][0, 0][0])  # 037454012.jpg

firstAnno = first['annorect']
print('dtype of first annorect:', firstAnno.dtype)  # [('scale', 'O'), ('objpos', 'O')]
print('type of first annorect objpos:', type(firstAnno['objpos']))
print('shape of first annorect objpos:', firstAnno['objpos'].shape)  # (1, 1)
print('dtype of first annorect objpos:', firstAnno['objpos'].dtype)  # object
print('first annorect objpos x,y: ({},{})'.format(firstAnno['objpos']['x'][0, 0][0], firstAnno['objpos']['y'][0, 0][0]))
#
#
# # x1, .y1, .x2, .y2
# secImg = content[0, 1]['image']
#
#
# print(type(version))
# print(version.shape)
# print(version.dtype)
# print(version[0, 0])






