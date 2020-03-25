import cv2
import numpy as np
from os.path import join


scores = np.array([2, 3, 1, 4, 9, 6, 7, 8, 5, 10], np.int32)
score_idx = np.argsort(scores)[::-1]
print('scores:', scores)
print('score_idx:', score_idx)

while len(score_idx) > 0:
    box_idx = score_idx[0]
    print('box_idx:', box_idx)
    to_delete = [0]
    for i, s in enumerate(score_idx):
        print('s:', s)
        if s == box_idx:
            continue
        try:
            if i % 2 == 3:
                to_delete.append(i)
                print('to_delete:', to_delete)
            # score_idx = np.delete(score_idx, i, 0)
            # print('score_idx:', score_idx)
        except:
            pass
    print('to_delete:', to_delete)
    score_idx = np.delete(score_idx, to_delete, 0)
    print('score_idx:', score_idx)





