"""嘴.py —— 发声通道。概念激活 + 身体声脑激活 → 扬声器。

人类语言进化的核心机制：
  特定的内部状态 → 特定的声音模式。
  概念层的激活模式就是内部状态——不同的概念组合产生不同的声音。
  反复发声 + 听觉反馈 → 声音模式和概念自然绑定（赫布学习）。

发声器官的物理：振动频率与振幅由概念激活和声脑激活共同决定。
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
    """发声器官。发声(概念激活, 声脑激活) 把内部状态转成真实声波。

    双输入驱动：
    - 概念激活（64维）：高层概念状态决定声音的"语义"——哪些概念活跃
    - 声脑激活（24维）：底层声脑状态决定声音的"物理"——振幅和基频

    映射规则：
    - 概念激活 → 频点选择（哪些频点被激活）
    - 声脑激活 → 振幅调制（每个频点的强度）
    - 概念激活的模式不同 → 声音模式不同 → 自然区分
    """

    def __init__(self, 声脑节点数=24, 概念节点数=64):
        self.声脑节点数 = 声脑节点数
        self.概念节点数 = 概念节点数
        self.离线 = not _SD_OK
        self._频点 = np.linspace(最低频, 最高频, 声脑节点数)
        self.最近发声拍 = -1

        # 概念→频点映射权重（赫布学习自然形成）
        # 初始小随机值——通过说听闭环自然学习
        # 当概念和特定频点反复共同激活时，连接增强（赫布学习）
        import random
        rng = random.Random(42)
        self.概念频点权重 = [
            [rng.uniform(-0.05, 0.05) for _ in range(声脑节点数)]
            for _ in range(概念节点数)
        ]

    def 发声(self, 概念激活=None, 声脑激活=None, 当前拍=0, 强制=False):
        """把概念激活和声脑激活转成真实声波。返回是否发了声。

        概念激活驱动频点选择——不同的概念组合 → 不同的声音模式。
        声脑激活驱动振幅调制——底层状态决定声音强度。
        """
        if self.离线:
            return False

        if 概念激活 is None and 声脑激活 is None:
            return False

        if 声脑激活 is not None:
            激活 = np.asarray(声脑激活, dtype=np.float64)
        else:
            激活 = np.zeros(self.声脑节点数)

        if 概念激活 is not None:
            概念 = np.asarray(概念激活, dtype=np.float64)
        else:
            概念 = np.zeros(self.概念节点数)

        if not 强制:
            if float(激活.mean()) <= 发声阈值 and float(np.max(np.abs(概念))) <= 0.3:
                return False

        t = np.linspace(0.0, 发声时长, int(采样率 * 发声时长), endpoint=False)
        波形 = np.zeros_like(t)

        n_声 = min(len(激活), self.声脑节点数)
        n_概念 = min(len(概念), self.概念节点数)

        for i in range(n_声):
            a = float(激活[i])
            if a <= 0.2:
                continue

            freq = float(self._频点[i])

            for j in range(n_概念):
                w = self.概念频点权重[j][i]
                if abs(w) > 0.01 and abs(概念[j]) > 0.1:
                    freq += w * 概念[j] * 50.0

            freq = max(最低频 * 0.5, min(最高频 * 2.0, freq))
            波形 += a * np.sin(2 * np.pi * freq * t)

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
