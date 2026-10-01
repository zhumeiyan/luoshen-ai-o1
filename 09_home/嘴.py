"""嘴.py —— 发声通道。身体声脑激活 → 扬声器。

发声器官的物理：振动频率与振幅由声脑激活决定，映射规则是器官结构。
不预设任何词、任何旋律——它想怎么叫，由它自己的激活决定。
它发出的声波会被自己的耳朵收回来——真实闭环。
设备离线就哑，宁可哑，不造假声。
"""

import numpy as np

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

采样率 = 22050
发声时长 = 0.3
发声阈值 = 0.15
最低频 = 80.0
最高频 = 400.0


class 嘴:
    """发声器官。发声(声脑激活) 把声脑激活转成真实声波播出去。

    声脑激活：24 节点激活列表（身体引擎脑1）。
    映射：激活 > 阈值的节点各对应一个基频（80-400Hz 均匀分布），
    正弦叠加，振幅 = 激活强度。零激活则不出声。
    """

    def __init__(self, 声脑节点数=24):
        self.声脑节点数 = 声脑节点数
        self.离线 = not _SD_OK
        self._频点 = np.linspace(最低频, 最高频, 声脑节点数)
        self.最近发声拍 = -1

    def 发声(self, 声脑激活, 当前拍=0, 强制=False):
        """把声脑激活转成真实声波。返回是否发了声。

        发声条件由激活本身决定：均值超过发声阈值才叫。
        """
        if self.离线 or 声脑激活 is None or len(声脑激活) == 0:
            return False
        激活 = np.asarray(声脑激活, dtype=np.float64)
        if not 强制 and float(激活.mean()) <= 发声阈值:
            return False

        t = np.linspace(0.0, 发声时长, int(采样率 * 发声时长), endpoint=False)
        波形 = np.zeros_like(t)
        n = min(len(激活), self.声脑节点数)
        for i in range(n):
            a = float(激活[i])
            if a > 0.2:
                波形 += a * np.sin(2 * np.pi * self._频点[i] * t)
        峰 = float(np.max(np.abs(波形)))
        if 峰 < 1e-4:
            return False
        波形 = 波形 / 峰 * min(0.6, 0.2 + 峰)

        try:
            sd.play(波形.astype(np.float32), 采样率)
            self.最近发声拍 = 当前拍
            return True
        except Exception:
            self.离线 = True
            return False

    def 等待播完(self):
        if not self.离线 and _SD_OK:
            try:
                sd.wait()
            except Exception:
                pass
