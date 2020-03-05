import cv2
import numpy as np
import time


class CaptureManager(object):
    '''操作视频流的控制类'''

    def __init__(self, capture, previewWindowManager = None, shouldMirrorPreview = False):
        '''
        初始化方法 \n
        capture -- VideoCapture类 \n
        previewWindowManager -- WindowManager类 \n
        shouldMirrorPreview -- 是否镜像
        '''
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        # 初始化各类成员变量
        self._channel = 0
        self._enteredFrame = False  # 是否进入帧处理
        self._frame = None
        self._imageFileName = None  # 保存截图位置
        self._videoFileName = None  # 保存视频位置
        self._videoEncoding = None  # 视频格式
        self._videoWriter = None  # 视频保存VideoWriter类
        # 估计FPS
        self._startTime = None
        self._frameElapsed = int(0)
        self._fpsEstimate = 0.0

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @frame.setter
    def frame(self, value):
        self._frame = value

    @property
    def isWritingImg(self):
        return self._imageFileName is not None

    @property
    def isWritingVideo(self):
        return self._videoFileName is not None

    def _writeVideo(self):
        '''保存视频的具体实现方法'''
        if not self.isWritingVideo:
            return
        # 如果还没有videoWriter，则初始化
        if self._videoWriter is None:
            # fps = self._capture.get(cv2.CAP_PROP_FPS)
            fps = None
            if fps is None or fps == 0.0:
                # 小于20帧的时候不保存
                if self._frameElapsed < 20:
                    pass
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFileName,
                                                self._videoEncoding, int(fps), size)
        else:
            self._videoWriter.write(self._frame)

    def enterFrame(self):
        '''从摄像头中读取帧，每调用一次仅尝试获取一帧数据'''
        # 首先检测是否设置并打开了摄像头
        assert self._capture is not None, \
            'You have not correctly set and open camera!'

        # 首先检测是否退出了上一帧的处理
        assert not self._enteredFrame, \
            'You have not exited from previous frame!'

        self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        '''在窗口中展示帧、保存帧至文件、清除帧数据，退出该帧操作'''
        # 获取帧数据
        if self.frame is None:
            self._enteredFrame = None
            return

        # 估算FPS
        if self._frameElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapse = time.time() - self._startTime
            self._fpsEstimate = self._frameElapsed / timeElapse
        self._frameElapsed += 1

        # 窗口中展示帧
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                frameToShow = np.fliplr(self._frame).copy()
            else:
                frameToShow = self._frame.copy()
            cv2.putText(frameToShow, 'FPS: {:.2f}'.format(self._fpsEstimate),
                        org=(0, 50), color=(0, 0, 0),
                        thickness=2, fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
            self.previewWindowManager.show(frameToShow)

        # 截屏保存
        if self.isWritingImg:
            cv2.imwrite(self._imageFileName, self._frame)
            print('截屏保存至: {}'.format(self._imageFileName))
            self._imageFileName = None

        # 视频保存
        if self.isWritingVideo:
            self._writeVideo()

        # 释放帧，重置各种flag
        self._frame = None
        self._enteredFrame = False

    def snapShot(self, fileName):
        '''截屏当前帧至文件'''
        self._imageFileName = fileName

    def startRecording(self, filename,
                       encoding=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')):
        '''开始保存视频'''
        self._videoFileName = filename
        self._videoEncoding = encoding

    def stopRecording(self):
        '''停止保存视频'''
        self._videoFileName =None
        self._videoEncoding = None
        self._videoWriter.release()
        self._videoWriter = None

    def releaseCapture(self):
        if self._capture is not None:
            if self.isWritingVideo:
                self.stopRecording()
            self._capture.release()


class WindowManager(object):
    '''控制窗口展示的控制类'''
    def __init__(self, windowName, keypressCallback = None):
        '''构造函数 \n
        windowName -- 窗口名称 \n
        keypressCallback -- 按键回调函数
        '''
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        '''创建该窗口'''
        cv2.namedWindow(self._windowName, cv2.WINDOW_NORMAL)
        self._isWindowCreated = True

    def show(self, frame):
        '''在该窗口中展示数据'''
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        '''销毁该窗口'''
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        '''处理该窗口的按键事件'''
        keyCode = cv2.waitKey(1)
        if self.keypressCallback is not None and keyCode != -1:
            self.keypressCallback(keyCode)



