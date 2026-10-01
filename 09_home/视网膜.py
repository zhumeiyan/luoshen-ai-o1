"""视网膜.py —— 眼睛。摄像头 → 光通道转译器。

只做物理亮度转译，像视网膜感光：零识别、零标注、零判断。
绝不生成任何动态数据——设备离线就失明，宁可失明，不喂假数据。
"""

import numpy as np

try:
    import cv2
    _CV_OK = True
except ImportError:
    _CV_OK = False


class 视网膜:
    """摄像头转译器。看() 返回 (光值, 光矩阵, 离线标记)。

    光值域 1-500（身体引擎光通道域）。
    光矩阵 8×64 灰度，仅留痕用，不做任何内容识别。
    """

    def __init__(self, 设备号=0, 宽=640, 高=480):
        self.设备号 = 设备号
        self.宽 = 宽
        self.高 = 高
        self._镜头 = None
        self.离线 = True
        if _CV_OK:
            self._镜头 = cv2.VideoCapture(设备号)
            if self._镜头.isOpened():
                self._镜头.set(cv2.CAP_PROP_FRAME_WIDTH, 宽)
                self._镜头.set(cv2.CAP_PROP_FRAME_HEIGHT, 高)
                self.离线 = False
        self.基线光 = 50.0

    def 看(self):
        """采一帧真实光。返回 (光值, 矩阵, 离线)。"""
        if self.离线 or self._镜头 is None:
            return self.基线光, None, True

        好读到, 帧 = self._镜头.read()
        if not 好读到 or 帧 is None:
            self.离线 = True
            self._释放()
            return self.基线光, None, True

        灰 = cv2.cvtColor(帧, cv2.COLOR_BGR2GRAY)
        小图 = cv2.resize(灰, (64, 8), interpolation=cv2.INTER_AREA)
        矩阵 = 小图.astype(np.float64)

        # 转译：摄像头自动曝光后的帧均值反映场景亮度。
        # 线性映射到身体光域 1-500（转译层压缩映射，量纲为工程估计）。
        均值 = float(矩阵.mean()) / 255.0
        光值 = 1.0 + 均值 * 499.0
        return 光值, 矩阵, False

    def _释放(self):
        if self._镜头 is not None:
            try:
                self._镜头.release()
            except Exception:
                pass
            self._镜头 = None

    def 关闭(self):
        self._释放()
