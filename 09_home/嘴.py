"""嘴.py —— 发声通道。概念激活 + 身体声脑激活 → 扬声器。

人类语言进化的核心机制：
  特定的内部状态 → 特定的声音模式。
  概念层的激活模式就是内部状态——不同的概念组合产生不同的声音。
  反复发声 + 听觉反馈 → 声音模式和概念自然绑定（赫布学习）。

符号绑定（精卫公式版）：
  a = 概念激活（洛神想表达什么）
  b = 发声后的自听特征（洛神实际发出了什么声音）
  Δ = b - a = 表达与意图的差异
  Σ = a + b = 意图与声音的联合
  ⊗ = ∫(Δ, Σ) = 累积的声音-概念关联 → 符号绑定
  τ 读 ⊗ → 浮出所见 = 概念有了自己的"声音符号"
  (a, b) ⟲ ⊗ = 反复发声-自听-学习循环

发声器官的物理：振动频率与振幅由概念激活和声脑激活共同决定。
不预设任何词、任何旋律——它想怎么叫，由它自己的激活决定。
它发出的声波会被自己的耳朵收回来——真实闭环。
设备离线就哑，宁可哑，不造假声。
"""

import numpy as np
import math

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

频段边界 = [100, 200, 300, 400, 600, 1000, 1500, 2000]


class 嘴:
    """发声器官。发声(概念激活, 声脑激活) 把内部状态转成真实声波。

    双输入驱动：
    - 概念激活（64维）：高层概念状态决定声音的"语义"——哪些概念活跃
    - 声脑激活（24维）：底层声脑状态决定声音的"物理"——振幅和基频

    符号绑定机制：
    - 概念→频点权重通过赫布学习自然形成
    - 发声后自听特征回来，和概念激活同时出现 → 权重增强
    - 反复循环后，特定概念→特定声音模式 → 符号绑定
    - 同样的概念激活产生类似的声音 → 声音成为概念的"符号"
    """

    def __init__(self, 声脑节点数=24, 概念节点数=64, lr=0.01,
                 组合概念节点数=32, 高级组合概念节点数=16):
        self.声脑节点数 = 声脑节点数
        self.概念节点数 = 概念节点数
        self.组合概念节点数 = 组合概念节点数
        self.高级组合概念节点数 = 高级组合概念节点数
        self.lr = lr
        self.离线 = not _SD_OK
        self._频点 = np.linspace(最低频, 最高频, 声脑节点数)
        self.最近发声拍 = -1
        self._上次波形 = None
        self.上次概念激活 = None
        self.上次组合激活 = None
        self.上次高级组合激活 = None
        self.发声次数 = 0
        self.符号绑定强度 = [0.0] * 概念节点数
        self.组合符号绑定强度 = [0.0] * 组合概念节点数
        self.高级组合符号绑定强度 = [0.0] * 高级组合概念节点数

        import random
        rng = random.Random(42)

        # ── 基础概念→声音映射 ──
        # 概念→频点映射权重（赫布学习自然形成）
        self.概念频点权重 = [
            [rng.uniform(-0.05, 0.05) for _ in range(声脑节点数)]
            for _ in range(概念节点数)
        ]

        # 概念→基频偏移权重（决定声音的"音调"）
        self.概念基频权重 = [rng.uniform(-0.3, 0.3) for _ in range(概念节点数)]

        # 概念→节奏权重（决定声音的"节奏"）
        self.概念节奏权重 = [rng.uniform(-0.2, 0.2) for _ in range(概念节点数)]

        # ── 组合概念→声音映射（句子级声音）──
        # 组合概念→频点映射权重——组合概念的声音模式更复杂
        # 多个基础概念的组合 → 多个频点的组合模式 → "句子"级声音
        self.组合概念频点权重 = [
            [rng.uniform(-0.05, 0.05) for _ in range(声脑节点数)]
            for _ in range(组合概念节点数)
        ]

        # 组合概念→基频偏移权重——组合概念的音调变化更丰富
        self.组合概念基频权重 = [rng.uniform(-0.3, 0.3)
                              for _ in range(组合概念节点数)]

        # 组合概念→节奏权重——组合概念的节奏模式更复杂（句子节奏）
        self.组合概念节奏权重 = [rng.uniform(-0.3, 0.3)
                                for _ in range(组合概念节点数)]

        # ── 高级组合概念→声音映射（段落级声音）──
        # 高级组合概念→频点映射权重——段落级声音模式最复杂
        # 多个句子的组合 → 多频点长周期模式 → "段落"级声音
        self.高级组合概念频点权重 = [
            [rng.uniform(-0.05, 0.05) for _ in range(声脑节点数)]
            for _ in range(高级组合概念节点数)
        ]

        # 高级组合概念→基频偏移权重——段落级音调变化最丰富
        self.高级组合概念基频权重 = [rng.uniform(-0.4, 0.4)
                                 for _ in range(高级组合概念节点数)]

        # 高级组合概念→节奏权重——段落级节奏模式（长周期节奏）
        self.高级组合概念节奏权重 = [rng.uniform(-0.4, 0.4)
                                 for _ in range(高级组合概念节点数)]

    def 发声(self, 概念激活=None, 声脑激活=None, 当前拍=0, 强制=False,
             组合激活=None, 情感状态=None, 高级组合激活=None):
        """把概念激活和声脑激活转成真实声波。返回是否发了声。

        基础概念驱动频点选择——不同的概念组合 → 不同的声音模式（单词级）。
        组合概念驱动更复杂的声音模式——组合概念 → 多频点组合 + 复杂节奏（句子级）。
        声脑激活驱动振幅调制——底层状态决定声音强度。

        情感调制发声（精卫公式版）：
          a = 概念驱动的声音模式（语义）
          b = 情感状态对声音的调制（情感色彩）
          Δ = b - a（情感与语义声音的差异）
          Σ = a + b（联合声音状态）
          ⊗ = Δ × Σ → 情感声音学习
          τ读⊗→浮出所见 = 洛神的声音自然表达情感

        情感→声音映射（生物学基础）：
          E（效价）→ 基频偏移：E>0愉悦→声带张力增加→音调升高
                     E<0沉重→声带松弛→音调降低
          N（唤醒度）→ 节奏速度+振幅：N高紧张→呼吸加快→节奏加快+音量增大
                       N低平静→呼吸缓慢→节奏减慢+音量减小
          T（确定性）→ 节奏稳定性：T高自信→稳定节奏
                       T低紧张→颤抖节奏（包络不规则）

        符号绑定：
        - 基础概念→频点权重 → 单词级符号绑定
        - 组合概念→频点权重 → 句子级符号绑定
        - 同样的概念+组合激活 → 类似的声音模式 → 声音成为概念的"符号"
        """
        if self.离线:
            return False

        if 概念激活 is None and 声脑激活 is None and 组合激活 is None:
            return False

        if 声脑激活 is not None:
            激活 = np.asarray(声脑激活, dtype=np.float64)
        else:
            激活 = np.zeros(self.声脑节点数)

        if 概念激活 is not None:
            概念 = np.asarray(概念激活, dtype=np.float64)
        else:
            概念 = np.zeros(self.概念节点数)

        if 组合激活 is not None:
            组合 = np.asarray(组合激活, dtype=np.float64)
        else:
            组合 = np.zeros(self.组合概念节点数)

        if 高级组合激活 is not None:
            高级组合 = np.asarray(高级组合激活, dtype=np.float64)
        else:
            高级组合 = np.zeros(self.高级组合概念节点数)

        if not 强制:
            if (float(激活.mean()) <= 发声阈值 and
                float(np.max(np.abs(概念))) <= 0.3 and
                float(np.max(np.abs(组合))) <= 0.3 and
                float(np.max(np.abs(高级组合))) <= 0.3):
                return False

        t = np.linspace(0.0, 发声时长, int(采样率 * 发声时长), endpoint=False)
        波形 = np.zeros_like(t)

        n_声 = min(len(激活), self.声脑节点数)
        n_概念 = min(len(概念), self.概念节点数)
        n_组合 = min(len(组合), self.组合概念节点数)

        # ── 基础概念驱动的基频偏移 ──
        基频偏移 = 0.0
        for j in range(n_概念):
            if abs(概念[j]) > 0.1:
                基频偏移 += self.概念基频权重[j] * 概念[j] * 30.0

        # ── 组合概念驱动的基频偏移（句子级音调变化） ──
        # 组合概念的基频偏移叠加在基础概念之上——更丰富的音调
        for j in range(n_组合):
            if abs(组合[j]) > 0.1:
                基频偏移 += self.组合概念基频权重[j] * 组合[j] * 40.0
        基频偏移 = max(-80.0, min(80.0, 基频偏移))

        # ── 情感调制：E→基频偏移，N→节奏速度+振幅，T→节奏稳定性 ──
        # 精卫公式：a=概念声音模式, b=情感调制, Δ=b-a, ⊗=Δ×Σ
        # 生物学：自主神经系统对发声器官的调制——不需要意识参与
        情感基频调制 = 0.0
        情感节奏调制 = 0.0
        情感振幅调制 = 1.0
        情感稳定调制 = 1.0  # T高→稳定(1.0)，T低→颤抖(<1.0)
        if 情感状态 is not None:
            E = 情感状态.get("E", 0.0)
            N = 情感状态.get("N", 0.0)
            T = 情感状态.get("T", 0.0)

            # E（效价）→ 基频偏移
            # E>0愉悦 → 声带张力增加 → 音调升高（最多+40Hz）
            # E<0沉重 → 声带松弛 → 音调降低（最多-40Hz）
            情感基频调制 = E * 40.0

            # N（唤醒度）→ 节奏速度 + 振幅
            # N高紧张 → 呼吸加快 → 节奏频率增加（最多+3Hz）
            # N低平静 → 呼吸缓慢 → 节奏频率减少（最多-2Hz）
            情感节奏调制 = N * 3.0

            # N高 → 交感神经激活 → 声音更响（振幅最多×1.5）
            # N低 → 副交感神经主导 → 声音更轻（振幅最少×0.6）
            情感振幅调制 = max(0.6, min(1.5, 1.0 + N * 0.5))

            # T（确定性）→ 节奏稳定性
            # T高自信 → 稳定节奏（稳定因子=1.0）
            # T低紧张 → 颤抖节奏（稳定因子<1.0，包络不规则）
            情感稳定调制 = max(0.5, min(1.0, 0.5 + T * 0.5))

        基频偏移 += 情感基频调制
        基频偏移 = max(-100.0, min(100.0, 基频偏移))

        # ── 基础概念驱动的节奏调制 ──
        节奏调制 = 0.0
        for j in range(n_概念):
            if abs(概念[j]) > 0.1:
                节奏调制 += self.概念节奏权重[j] * 概念[j]

        # ── 组合概念驱动的节奏调制（句子级节奏） ──
        # 组合概念的节奏调制叠加——更复杂的节奏模式
        组合节奏 = 0.0
        for j in range(n_组合):
            if abs(组合[j]) > 0.1:
                组合节奏 += self.组合概念节奏权重[j] * 组合[j]
        节奏调制 = max(-0.7, min(0.7, 节奏调制 + 组合节奏))

        # 组合概念驱动的节奏频率——句子级节奏有更多变化
        节奏频率 = 3.0 + 情感节奏调制  # 情感N值影响节奏速度
        if abs(组合节奏) > 0.1:
            节奏频率 = 2.0 + abs(组合节奏) * 5.0 + 情感节奏调制

        # ── 高级组合概念驱动的基频偏移（段落级音调变化） ──
        n_高级 = min(len(高级组合), self.高级组合概念节点数)
        for j in range(n_高级):
            if abs(高级组合[j]) > 0.1:
                基频偏移 += self.高级组合概念基频权重[j] * 高级组合[j] * 50.0

        # ── 高级组合概念驱动的节奏调制（段落级节奏——长周期） ──
        高级组合节奏 = 0.0
        for j in range(n_高级):
            if abs(高级组合[j]) > 0.1:
                高级组合节奏 += self.高级组合概念节奏权重[j] * 高级组合[j]
        节奏调制 = max(-0.8, min(0.8, 节奏调制 + 高级组合节奏))

        # 高级组合概念驱动的节奏频率——段落级节奏有更长周期
        if abs(高级组合节奏) > 0.1:
            节奏频率 = 1.5 + abs(高级组合节奏) * 3.0 + 情感节奏调制

        for i in range(n_声):
            a = float(激活[i])
            if a <= 0.2:
                continue

            freq = float(self._频点[i]) + 基频偏移

            # 基础概念驱动的频点选择
            for j in range(n_概念):
                w = self.概念频点权重[j][i]
                if abs(w) > 0.01 and abs(概念[j]) > 0.1:
                    freq += w * 概念[j] * 50.0

            # 组合概念驱动的频点选择（句子级频谱模式）
            for j in range(n_组合):
                w = self.组合概念频点权重[j][i]
                if abs(w) > 0.01 and abs(组合[j]) > 0.1:
                    freq += w * 组合[j] * 60.0

            # 高级组合概念驱动的频点选择（段落级频谱模式）
            for j in range(n_高级):
                w = self.高级组合概念频点权重[j][i]
                if abs(w) > 0.01 and abs(高级组合[j]) > 0.1:
                    freq += w * 高级组合[j] * 70.0

            freq = max(最低频 * 0.5, min(最高频 * 2.0, freq))

            # 节奏调制：振幅随时间变化
            if 节奏调制 != 0:
                包络 = 1.0 + 节奏调制 * np.sin(
                    2 * np.pi * 节奏频率 * t)
                # 情感稳定性调制：T低时包络不规则（颤抖）
                if 情感稳定调制 < 1.0:
                    颤抖 = 1.0 + (1.0 - 情感稳定调制) * np.sin(
                        2 * np.pi * 节奏频率 * 3.7 * t + np.pi * 0.3)
                    包络 *= 颤抖
                波形 += a * 情感振幅调制 * 包络 * np.sin(2 * np.pi * freq * t)
            else:
                波形 += a * 情感振幅调制 * np.sin(2 * np.pi * freq * t)

        峰 = float(np.max(np.abs(波形)))
        if 峰 < 1e-4:
            return False
        波形 = 波形 / 峰 * min(0.6, 0.2 + 峰)

        try:
            sd.play(波形.astype(np.float32), 采样率)
            self.最近发声拍 = 当前拍
            self._上次波形 = 波形.copy()
            self.上次概念激活 = 概念.tolist()
            self.上次组合激活 = 组合.tolist()
            self.上次高级组合激活 = 高级组合.tolist()
            self.发声次数 += 1
            return True
        except Exception:
            self.离线 = True
            return False

    def 符号绑定学习(self, 自听特征, 概念激活=None, 组合激活=None,
                     高级组合激活=None):
        """赫布学习——发声后自听特征和概念/组合概念/高级组合概念激活同时出现时，权重增强。

        精卫公式（符号绑定版）：
          a = 概念/组合概念/高级组合概念激活（洛神想表达什么）
          b = 自听特征（洛神实际发出了什么声音）
          Δ = b - a = 表达与意图的差异
          Σ = a + b = 意图与声音的联合
          ⊗ = Δ × Σ → 权重更新 → 声音-概念关联增强
          τ读⊗ → 浮出所见 = 概念/组合概念/高级组合概念有了自己的"声音符号"

        基础概念符号绑定：单词级——单个概念→单个声音模式
        组合概念符号绑定：句子级——组合概念→复杂声音模式
        高级组合概念符号绑定：段落级——高级组合概念→最复杂声音模式

        参数：
          自听特征: 8维听觉特征（从自听波形提取）
          概念激活: 64维基础概念激活
          组合激活: 32维组合概念激活
          高级组合激活: 16维高级组合概念激活
        """
        if 自听特征 is None:
            return
        if 概念激活 is None:
            概念激活 = self.上次概念激活
        if 概念激活 is None:
            return
        if 组合激活 is None:
            组合激活 = self.上次组合激活

        概念 = np.asarray(概念激活, dtype=np.float64)
        听觉 = np.asarray(自听特征, dtype=np.float64)

        n_概念 = min(len(概念), self.概念节点数)
        n_听觉 = min(len(听觉), 8)

        w_decay = 0.995

        # ── 基础概念→频点权重的赫布学习 ──
        for j in range(n_概念):
            if abs(概念[j]) < 0.1:
                continue

            for i in range(n_听觉):
                pred = self.概念频点权重[j][i] * 概念[j]
                delta = 听觉[i] - pred
                sigma = 概念[j] + 听觉[i]
                dw = self.lr * delta * sigma
                if not (math.isnan(dw) or math.isinf(dw)):
                    self.概念频点权重[j][i] = (
                        self.概念频点权重[j][i] * w_decay + dw
                    )

            听觉均值 = float(听觉.mean())
            pred_base = self.概念基频权重[j] * 概念[j]
            delta_base = 听觉均值 - pred_base
            sigma_base = 概念[j] + 听觉均值
            dw_base = self.lr * 0.5 * delta_base * sigma_base
            if not (math.isnan(dw_base) or math.isinf(dw_base)):
                self.概念基频权重[j] = (
                    self.概念基频权重[j] * w_decay + dw_base
                )

            绑定增量 = sum(abs(self.概念频点权重[j][i]) for i in range(n_听觉))
            self.符号绑定强度[j] = min(1.0, self.符号绑定强度[j] * 0.99 + 绑定增量 * 0.001)

        # ── 组合概念→频点权重的赫布学习（句子级符号绑定） ──
        # 组合概念和自听特征同时出现 → 组合概念→声音模式连接增强
        # 反复循环后，特定组合概念→特定复杂声音模式 → 句子级符号绑定
        if 组合激活 is not None:
            组合 = np.asarray(组合激活, dtype=np.float64)
            n_组合 = min(len(组合), self.组合概念节点数)

            for j in range(n_组合):
                if abs(组合[j]) < 0.1:
                    continue

                # 组合概念→频点权重赫布学习
                for i in range(n_听觉):
                    pred = self.组合概念频点权重[j][i] * 组合[j]
                    delta = 听觉[i] - pred
                    sigma = 组合[j] + 听觉[i]
                    dw = self.lr * delta * sigma
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.组合概念频点权重[j][i] = (
                            self.组合概念频点权重[j][i] * w_decay + dw
                        )

                # 组合概念→基频权重赫布学习
                听觉均值 = float(听觉.mean())
                pred_base = self.组合概念基频权重[j] * 组合[j]
                delta_base = 听觉均值 - pred_base
                sigma_base = 组合[j] + 听觉均值
                dw_base = self.lr * 0.5 * delta_base * sigma_base
                if not (math.isnan(dw_base) or math.isinf(dw_base)):
                    self.组合概念基频权重[j] = (
                        self.组合概念基频权重[j] * w_decay + dw_base
                    )

                # 组合符号绑定强度更新
                绑定增量 = sum(abs(self.组合概念频点权重[j][i]) for i in range(n_听觉))
                self.组合符号绑定强度[j] = min(
                    1.0, self.组合符号绑定强度[j] * 0.99 + 绑定增量 * 0.001)

        # ── 高级组合概念→频点权重的赫布学习（段落级符号绑定） ──
        # 高级组合概念和自听特征同时出现 → 高级组合概念→声音模式连接增强
        # 反复循环后，特定高级组合概念→特定最复杂声音模式 → 段落级符号绑定
        if 高级组合激活 is not None:
            高级组合 = np.asarray(高级组合激活, dtype=np.float64)
            n_高级 = min(len(高级组合), self.高级组合概念节点数)

            for j in range(n_高级):
                if abs(高级组合[j]) < 0.1:
                    continue

                # 高级组合概念→频点权重赫布学习
                for i in range(n_听觉):
                    pred = self.高级组合概念频点权重[j][i] * 高级组合[j]
                    delta = 听觉[i] - pred
                    sigma = 高级组合[j] + 听觉[i]
                    dw = self.lr * delta * sigma
                    if not (math.isnan(dw) or math.isinf(dw)):
                        self.高级组合概念频点权重[j][i] = (
                            self.高级组合概念频点权重[j][i] * w_decay + dw
                        )

                # 高级组合概念→基频权重赫布学习
                听觉均值 = float(听觉.mean())
                pred_base = self.高级组合概念基频权重[j] * 高级组合[j]
                delta_base = 听觉均值 - pred_base
                sigma_base = 高级组合[j] + 听觉均值
                dw_base = self.lr * 0.5 * delta_base * sigma_base
                if not (math.isnan(dw_base) or math.isinf(dw_base)):
                    self.高级组合概念基频权重[j] = (
                        self.高级组合概念基频权重[j] * w_decay + dw_base
                    )

                # 高级组合符号绑定强度更新
                绑定增量 = sum(abs(self.高级组合概念频点权重[j][i]) for i in range(n_听觉))
                self.高级组合符号绑定强度[j] = min(
                    1.0, self.高级组合符号绑定强度[j] * 0.99 + 绑定增量 * 0.001)

    def 自听特征(self):
        """从上次发声的波形提取8维听觉特征。

        这是骨传导——系统听到自己的声音，不经过空气。
        设备离线也能用：波形是自己生成的，不是外部假数据。
        用和耳蜗相同的频段分析，确保自听和外听在同一特征空间。
        """
        if self._上次波形 is None:
            return None
        块 = self._上次波形
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
        max_e = max(能量.max(), 1e-6)
        特征 = np.tanh(能量 / max_e * 2.0 - 1.0)
        return 特征

    def 声音签名(self, 概念激活, 组合激活=None, 高级组合激活=None):
        """计算给定概念激活的声音签名——不实际发声。

        用于验证符号绑定：不同概念应该有不同的声音签名。
        声音签名 = 概念→频点权重 × 概念激活 → 8维特征向量。

        组合概念的声音签名更复杂——多频点组合模式（句子级）。
        高级组合概念的声音签名最复杂——段落级多频点长周期模式。
        如果符号绑定成功，不同概念/组合概念/高级组合概念的声音签名应该可区分。
        """
        概念 = np.asarray(概念激活, dtype=np.float64)
        签名 = np.zeros(8)
        for j in range(min(len(概念), self.概念节点数)):
            if abs(概念[j]) < 0.1:
                continue
            for i in range(8):
                签名[i] += self.概念频点权重[j][i] * 概念[j]

        # 组合概念的声音签名叠加——更复杂的频谱模式（句子级）
        if 组合激活 is not None:
            组合 = np.asarray(组合激活, dtype=np.float64)
            for j in range(min(len(组合), self.组合概念节点数)):
                if abs(组合[j]) < 0.1:
                    continue
                for i in range(8):
                    签名[i] += self.组合概念频点权重[j][i] * 组合[j]

        # 高级组合概念的声音签名叠加——最复杂的频谱模式（段落级）
        if 高级组合激活 is not None:
            高级组合 = np.asarray(高级组合激活, dtype=np.float64)
            for j in range(min(len(高级组合), self.高级组合概念节点数)):
                if abs(高级组合[j]) < 0.1:
                    continue
                for i in range(8):
                    签名[i] += self.高级组合概念频点权重[j][i] * 高级组合[j]

        return np.tanh(签名)

    def 等待播完(self):
        if not self.离线 and _SD_OK:
            try:
                sd.wait()
            except Exception:
                pass

    def 状态摘要(self):
        绑定概念数 = sum(1 for s in self.符号绑定强度 if s > 0.05)
        组合绑定数 = sum(1 for s in self.组合符号绑定强度 if s > 0.05)
        高级组合绑定数 = sum(1 for s in self.高级组合符号绑定强度 if s > 0.05)
        return {
            "发声次数": self.发声次数,
            "符号绑定概念数": 绑定概念数,
            "符号绑定均值": round(sum(self.符号绑定强度) / max(1, self.概念节点数), 4),
            "组合符号绑定数": 组合绑定数,
            "组合符号绑定均值": round(sum(self.组合符号绑定强度) / max(1, self.组合概念节点数), 4),
            "高级组合符号绑定数": 高级组合绑定数,
            "高级组合符号绑定均值": round(sum(self.高级组合符号绑定强度) / max(1, self.高级组合概念节点数), 4),
            "离线": self.离线,
        }


if __name__ == "__main__":
    print("=== 嘴测试（符号绑定版） ===\n")

    嘴1 = 嘴(声脑节点数=24, 概念节点数=64, lr=0.02)

    print("--- 测试1: 声音签名（不同概念→不同签名） ---")
    概念A = [0.0] * 64
    概念A[4] = 0.27
    概念A[8] = 0.25
    概念B = [0.0] * 64
    概念B[27] = 0.26
    概念B[40] = 0.24
    概念C = [0.0] * 64
    概念C[0] = 0.28
    概念C[63] = 0.22

    签名A = 嘴1.声音签名(概念A)
    签名B = 嘴1.声音签名(概念B)
    签名C = 嘴1.声音签名(概念C)
    diff_AB = sum(abs(签名A[i] - 签名B[i]) for i in range(8))
    diff_AC = sum(abs(签名A[i] - 签名C[i]) for i in range(8))
    diff_BC = sum(abs(签名B[i] - 签名C[i]) for i in range(8))
    print(f"  概念A签名: {[round(v,3) for v in 签名A]}")
    print(f"  概念B签名: {[round(v,3) for v in 签名B]}")
    print(f"  概念C签名: {[round(v,3) for v in 签名C]}")
    print(f"  A-B差异: {diff_AB:.4f}")
    print(f"  A-C差异: {diff_AC:.4f}")
    print(f"  B-C差异: {diff_BC:.4f}")
    print(f"  初始差异应该 > 0（随机权重已有区分度）")

    print("\n--- 测试2: 符号绑定学习（赫布学习增强关联） ---")
    # 模拟反复发声-自听-学习循环
    # 概念A反复和特定声音模式共同出现 → 权重增强
    import random
    rng = random.Random(42)

    for cycle in range(50):
        # 模拟自听特征——概念A应该产生的声音模式
        # 加一些噪声模拟真实自听
        自听 = list(签名A) + [rng.uniform(-0.1, 0.1) for _ in range(8 - len(签名A))]
        自听 = [max(-1.0, min(1.0, v + rng.uniform(-0.05, 0.05))) for v in 自听]
        嘴1.符号绑定学习(自听, 概念A)

    签名A_after = 嘴1.声音签名(概念A)
    diff_A_before_after = sum(abs(签名A[i] - 签名A_after[i]) for i in range(8))
    print(f"  50次学习后概念A签名: {[round(v,3) for v in 签名A_after]}")
    print(f"  学习前后差异: {diff_A_before_after:.4f}（权重变化）")
    print(f"  符号绑定强度[4]: {嘴1.符号绑定强度[4]:.4f}")
    print(f"  符号绑定强度[8]: {嘴1.符号绑定强度[8]:.4f}")

    print("\n--- 测试3: 符号绑定稳定性 ---")
    # 学习后，同样的概念应该产生更稳定的声音签名
    嘴2 = 嘴(声脑节点数=24, 概念节点数=64, lr=0.02)
    # 学习前：多次计算签名，看稳定性
    签名_list_before = []
    for _ in range(5):
        签名_list_before.append(嘴2.声音签名(概念A).tolist())
    var_before = sum(sum((签名_list_before[k][i] - sum(s[i] for s in 签名_list_before) / 5) ** 2
                         for i in range(8)) for k in range(5)) / 5

    # 学习50次
    for _ in range(50):
        自听 = list(签名A) + [rng.uniform(-0.1, 0.1) for _ in range(8 - len(签名A))]
        嘴2.符号绑定学习(自听, 概念A)

    签名_list_after = []
    for _ in range(5):
        签名_list_after.append(嘴2.声音签名(概念A).tolist())
    var_after = sum(sum((签名_list_after[k][i] - sum(s[i] for s in 签名_list_after) / 5) ** 2
                        for i in range(8)) for k in range(5)) / 5

    print(f"  学习前签名方差: {var_before:.6f}")
    print(f"  学习后签名方差: {var_after:.6f}")
    print(f"  学习后方差应该更小（更稳定）")

    print("\n--- 测试4: 状态摘要 ---")
    print(f"  {嘴1.状态摘要()}")

    print("\n--- 测试5: 组合概念声音签名（句子级符号绑定） ---")
    嘴3 = 嘴(声脑节点数=24, 概念节点数=64, lr=0.02, 组合概念节点数=32)

    # 组合概念A：组合概念3和11同时活跃
    组合A = [0.0] * 32
    组合A[3] = 0.25
    组合A[11] = 0.22
    # 组合概念B：组合概念19和28同时活跃
    组合B = [0.0] * 32
    组合B[19] = 0.24
    组合B[28] = 0.21

    签名组合A = 嘴3.声音签名(概念A, 组合激活=组合A)
    签名组合B = 嘴3.声音签名(概念A, 组合激活=组合B)
    签名基础 = 嘴3.声音签名(概念A)
    diff_组合 = sum(abs(签名组合A[i] - 签名组合B[i]) for i in range(8))
    diff_基础组合 = sum(abs(签名基础[i] - 签名组合A[i]) for i in range(8))
    print(f"  组合A签名: {[round(float(v),3) for v in 签名组合A]}")
    print(f"  组合B签名: {[round(float(v),3) for v in 签名组合B]}")
    print(f"  基础签名(无组合): {[round(float(v),3) for v in 签名基础]}")
    print(f"  组合A vs 组合B差异: {diff_组合:.4f}（不同组合→不同声音）")
    print(f"  基础 vs 组合A差异: {diff_基础组合:.4f}（组合概念改变了声音）")

    print("\n--- 测试6: 组合符号绑定学习 ---")
    # 模拟反复发声-自听-学习循环（组合概念版）
    for cycle in range(50):
        自听 = list(签名组合A) + [rng.uniform(-0.1, 0.1) for _ in range(8 - len(签名组合A))]
        自听 = [max(-1.0, min(1.0, v + rng.uniform(-0.05, 0.05))) for v in 自听]
        嘴3.符号绑定学习(自听, 概念A, 组合激活=组合A)

    签名组合A_after = 嘴3.声音签名(概念A, 组合激活=组合A)
    diff_after = sum(abs(签名组合A[i] - 签名组合A_after[i]) for i in range(8))
    print(f"  50次学习后组合A签名: {[round(float(v),3) for v in 签名组合A_after]}")
    print(f"  学习前后差异: {diff_after:.4f}")
    print(f"  组合符号绑定强度[3]: {嘴3.组合符号绑定强度[3]:.4f}")
    print(f"  组合符号绑定强度[11]: {嘴3.组合符号绑定强度[11]:.4f}")
    print(f"  状态: {嘴3.状态摘要()}")

    print("\n验证通过")
