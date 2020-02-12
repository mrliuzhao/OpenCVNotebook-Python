import cv2
import math

cap = cv2.VideoCapture(r'.\resources\WOTLK_WEB_1280_Xvid_EN_ESRB.avi')
width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("width:", width, ';height:', height)
print("channel:", cap.get(cv2.CAP_PROP_CHANNEL))
print("Frame Count:", frameCount)
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)
timeElapse = int(math.floor(1000/cap.get(cv2.CAP_PROP_FPS)))


# 回调函数，x表示滑块的位置，本例暂不使用
def setTrackBar(x):
    cap.set(cv2.CAP_PROP_POS_FRAMES, x)
    pass


cv2.createTrackbar('Time', 'Video Player', 0, frameCount, setTrackBar)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('Return FALSE')
        break

    cv2.imshow("Video Player", frame)
    cv2.setTrackbarPos('Time', 'Video Player', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    if cv2.waitKey(timeElapse) == ord('q'):
        break

cv2.destroyAllWindows()



