import cv2
import os
from Cameo.managers import CaptureManager as CapMan
from Cameo.managers import WindowManager as WinMan
import Cameo.filters as fts
import Cameo.utils as ut


class Cameo(object):
    '''Cameo项目主类'''

    def __init__(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOCUS, 1)
        self._windowManager = WinMan('Cameo', self.onKeyPress)
        self._captureManager = CapMan(cap, self._windowManager, True)
        self.__imgFilter = fts.EmbossFilter()

    def run(self):
        '''主循环'''
        eye_xml = cv2.CascadeClassifier(r"C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_eye.xml")
        path = os.path.abspath('.')
        path = os.path.join(path, 'resources', 'heart.png')
        print(path)
        # hrt = cv2.imread(r"D:\PythonWorkspace\OpenCVNotebook-Python\resources\heart.png",
        #                  cv2.IMREAD_UNCHANGED)
        hrt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        hh, hw = hrt.shape[:2]
        b, g, r, a = cv2.split(hrt)
        hrt = cv2.merge((b, g, r))
        hrt = cv2.bitwise_and(hrt, hrt, mask=a)
        b, g, r = cv2.split(hrt)
        # ret, mask = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)
        alphaInv = cv2.bitwise_not(r)

        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            # TODO: 在此可以对帧进行各种操作
            # fts.strokeEdges(frame, frame)
            # self.__imgFilter.apply(frame, frame)

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # eyes = eye_xml.detectMultiScale(gray, 1.3, 5)
            # for (x, y, w, h) in eyes:
            #     cv2.rectangle(frame, (x, y), (x + w, y + h),
            #                   color=(0, 255, 255), thickness=2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = eye_xml.detectMultiScale(gray, 1.3, 5)
            for eyeRect in eyes:
                ut.copyRect(hrt, frame, dstRect=eyeRect, mask=r, lockRatio=True)

            # for (x, y, w, h) in eyes:
            #     scale = min(w / hw, h / hh)
            #     if scale > 1:
            #         scHrt = cv2.resize(hrt, None, fx=scale, fy=scale,
            #                            interpolation=cv2.INTER_CUBIC)
            #         scMask = cv2.resize(alphaInv, None, fx=scale, fy=scale,
            #                             interpolation=cv2.INTER_CUBIC)
            #     else:
            #         scHrt = cv2.resize(hrt, None, fx=scale, fy=scale,
            #                            interpolation=cv2.INTER_AREA)
            #         scMask = cv2.resize(alphaInv, None, fx=scale, fy=scale,
            #                             interpolation=cv2.INTER_AREA)
            #     xEnd = int(round(x + hw * scale))
            #     yEnd = int(round(y + hh * scale))
            #     roi = frame[y:yEnd, x:xEnd, :]
            #     rw, rh, c = roi.shape
            #     mw, mh = scMask.shape
            #     if rw == mw and rh == mh:
            #         roi = cv2.bitwise_and(roi, roi, mask=scMask)
            #         sub = cv2.add(roi, scHrt)
            #         frame[y:yEnd, x:xEnd, :] = sub
            #     else:
            #         print('roi shape, mask shape: ({},{})  ({},{})'.format(rw, rh, mw, mh))

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeyPress(self, keyCode):
        '''定义按键事件回调函数'''
        if keyCode == 32:  # 空格 -- 截屏
            self._captureManager.snapShot('snapshot.jpg')
        elif keyCode == 9:  # tab -- 开启/关闭视频保存
            if not self._captureManager.isWritingVideo:
                self._captureManager.startRecording('record.avi')
            else:
                self._captureManager.stopRecording()
        elif keyCode == 27:  # Esc -- 关闭
            self._captureManager.releaseCapture()
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    Cameo().run()


