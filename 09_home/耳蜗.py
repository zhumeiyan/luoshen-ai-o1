"""耳蜗.py —— 耳朵。麦克风 → 声通道转译器 + 听觉特征提取。

人类听觉系统的两个功能：
1. 声压物理转译（耳蜗毛细胞）：零语音识别、零转文字、零判断
2. 听觉特征提取（听皮层A1）：从声波中提取频谱特征

说听闭环的关键：
  发声 → 听回自己的声音 → 对比（预测误差）
  特定的声音模式和特定的概念反复共同出现 → 自然绑定（赫布学习）
  这是语言涌现的基础——声音模式成为概念的"符号"

绝不生成任何动态数据——设备离线就失聪，宁可失聪，不喂假数据。
"""

import threading
import numpy as np

try:
    import sounddevice as sd
    _SD_OK = True
except ImportError:
    _SD_OK = False

采样率 = 16000
块长 = 2048
频段边界 = [100, 200, 300, 400, 600, 1000, 1500, 2000]


class 耳蜗:
    """麦克风转译器。听() 返回 (声值, 声矩阵, 离线标记, 听觉特征)。

    声值域 25-75（身体引擎声通道域，dB SPL 工程估计）。
    声矩阵 8 频段 × 64 时间片滚动能量，仅留痕用，不做语音识别。
    听觉特征 8维向量——频谱特征，可用于激活概念层（说听闭环的输入端）。
    """

    def __init__(self, 设备号=None):
        self.设备号 = 设备号
        self.离线 = True
        self._队列 = []
        self._锁 = threading.Lock()
        self._流 = None
        self._历史 = np.zeros((8, 64))
        self._时间片 = 0
        if _SD_OK:
            try:
                self._流 = sd.InputStream(
                    samplerate=采样率, channels=1, dtype="float32",
                    blocksize=块长, device=设备号, callback=self._回调,
                )
                self._流.start()
                self.离线 = False
            except Exception:
                self._流 = None
                self.离线 = True
        self.基线声 = 45.0
        self._上次特征 = np.zeros(8)

    def _回调(self, 数据, 帧数, 时间, 状态):
        with self._锁:
            self._队列.append(数据[:, 0].copy())
            if len(self._队列) > 8:
                self._队列.pop(0)

    def 听(self):
        """采一片真实声。返回 (声值, 矩阵, 离线, 听觉特征)。

        听觉特征：8维频谱特征向量，归一化到[-1,1]。
        可用于激活概念层——说听闭环的输入端。
        """
        if self.离线 or self._流 is None:
            return self.基线声, self._历史.copy(), True, self._上次特征

        with self._锁:
            块 = self._队列[-1] if self._队列 else None

        if 块 is None or len(块) == 0:
            return self.基线声, self._历史.copy(), False, self._上次特征

        谱 = np.abs(np.fft.rfft(块))
        频点 = np.fft.rfftfreq(len(块), 1.0 / 采样率)
        能量 = np.zeros(8)
        上 = 0.0
        for i, 边界 in enumerate(频段边界):
            带 = 谱[(频点 > 上) & (频点 <= 边界)]
            能量[i] = float(带.mean()) if len(带) > 0 else 0.0
            上 = 边界
        高带 = 谱[频点 > 上]
        能量[7] = float(高带.mean()) if len(高带) > 0 else 0.0

        self._历史 = np.roll(self._历史, -1, axis=1)
        self._历史[:, -1] = 能量
        self._时间片 += 1

        均方根 = float(np.sqrt(np.mean(块 ** 2)))
        相对dB = 20.0 * np.log10(max(均方根, 1e-6))
        声值 = 25.0 + (min(max(相对dB, -60.0), 40.0) + 60.0) / 100.0 * 50.0

        max_e = max(能量.max(), 1e-6)
        特征 = np.tanh(能量 / max_e * 2.0 - 1.0)
        self._上次特征 = 特征

        return 声值, self._历史.copy(), False, 特征

    def 关闭(self):
        if self._流 is not None:
            try:
                self._流.stop()
                self._流.close()
            except Exception:
                pass
            self._流 = None
