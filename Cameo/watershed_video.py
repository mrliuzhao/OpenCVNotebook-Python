import numpy as np
import cv2

cur_marker = 1


def mouse_event(event, x, y, flags, param):
    global markers, cur_marker

    if event == cv2.EVENT_LBUTTONUP:
        print('鼠标位置：（{}, {}）'.format(x, y))
        print('cur_marker：', cur_marker)
        markers[x, y] = cur_marker


if __name__ == '__main__':
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    markers = np.zeros((h, w), np.int32)
    colors = np.int32(list(np.ndindex(2, 2, 2))) * 255

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('display', mouse_event)
    cv2.namedWindow('watershed', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 图片二值化，物体为白色
        ret, imgBi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow('display', cv2.hconcat((frame, cv2.cvtColor(imgBi, cv2.COLOR_GRAY2BGR))))

        cv2.watershed(frame, markers)

        overlay = colors[np.maximum(markers, 0)]

        vis = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)

        # cv2.imshow('display', frame)
        cv2.imshow('watershed', vis)

        keyCode = cv2.waitKey(1)
        if keyCode == 27:
            break
        if keyCode == ord('r'):
            markers = np.zeros((h, w), np.int32)
        if ord('1') <= keyCode <= ord('7'):
            cur_marker = keyCode - ord('0')
            print('marker: ', cur_marker)

    cap.release()
    cv2.destroyAllWindows()







