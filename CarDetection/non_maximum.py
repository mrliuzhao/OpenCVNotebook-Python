# import the necessary packages
import numpy as np


def area(box):
    return (abs(box[2] - box[0])) * (abs(box[3] - box[1]))


def overlaps(a, b, thresh=0.5):
    # print("checking overlap ")
    # print(a, b)
    x1 = np.maximum(a[0], b[0])
    x2 = np.minimum(a[2], b[2])
    y1 = np.maximum(a[1], b[1])
    y2 = np.minimum(a[3], b[3])
    intersect = float(area([x1, y1, x2, y2]))
    return intersect / np.minimum(area(a), area(b)) >= thresh


def is_inside(rec1, rec2):
    def inside(a, b):
        if (a[0] >= b[0]) and (a[2] <= b[0]):
            return (a[1] >= b[1]) and (a[3] <= b[3])
        else:
            return False

    return inside(rec1, rec2) or inside(rec2, rec1)


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh = 0.5):
    '''
    非极大值抑制

    :param boxes:
    :param overlapThresh:
    :return:
    '''

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    scores = boxes[:, 4]  # 分数数组
    score_idx = np.argsort(scores)[::-1]  # 排序后分数数组下标，从大到小

    boxes_res = []
    # box_to_delete = []
    while len(score_idx) > 0:
        # box_idx = score_idx[0]
        score_to_delete = [0]
        score_temp = []
        for i, s in enumerate(score_idx):
            if i == 0:
                score_temp.append(scores[s])
                boxtemp = boxes[s][:4]
                # box_to_delete.append(s)
                continue
            try:
                # 若重叠超过一定限度，对重叠矩形进行操作
                if overlaps(boxtemp, boxes[s], overlapThresh):
                    # box_to_delete.append(s)
                    score_temp.append(scores[s])
                    # 取并集的最小外接矩形
                    # x_min = min(boxtemp[0], boxes[s][0])
                    # x_max = max(boxtemp[2], boxes[s][2])
                    # y_min = min(boxtemp[1], boxes[s][1])
                    # y_max = max(boxtemp[3], boxes[s][3])
                    # boxtemp = (x_min, y_min, x_max, y_max)
                    # 保留面积较大的那个
                    if area(boxes[s]) > area(boxtemp):
                        boxtemp = boxes[s][:4]
                    score_to_delete.append(i)
                    # score_idx = np.delete(score_idx, [s], 0)
            except:
                pass
        score_idx = np.delete(score_idx, score_to_delete, 0)
        x1, y1, x2, y2 = boxtemp
        boxes_res.append((x1, y1, x2, y2, np.mean(score_temp)))
    # boxes = np.delete(boxes, box_to_delete, 0)

    return boxes_res


"""
  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  # initialize the list of picked indexes 
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]

  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))

  # return only the bounding boxes that were picked using the
  # integer data type
  return boxes[pick].astype("int")
"""
