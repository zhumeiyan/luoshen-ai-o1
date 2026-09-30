"""发声矩阵 v0.1 — 帧→矩阵→入痕（待核）。

规格来源：言语声学 / 语音产生。

发声是听觉的反向：听觉管"听"，发声管"说"。四样规格 + 共振峰：
  基频：60–500 Hz（成年男声~85-180Hz，女声~165-255Hz；极端可到60-500）
  声压：60–90 dB SPL（日常对话~60-70，喊叫~80-90）
  共振峰：F1 ~250-1000 Hz, F2 ~700-2500 Hz, F3 ~1500-3500 Hz
  分辨率：频率1–2 Hz，强度0.5–1 dB
  采样频率：16–48 kHz（语音采集的工程标准）

矩阵结构：8 行（声道分区：声带/咽腔/鼻腔/口腔/唇/舌/齿/下颌）
× 64 列（时间帧）。

纪律：
  - 不做语音识别/合成/情感推理
  - 只做"帧→声学参数映射→入痕"
  - 待升 [事实] 需哨兵核

实测说明：测试尺寸，设计值按声道具体标定。
"""

# ── 发声声学规格 ──
规格 = {
    "感官":         "发声",
    "子模态":       ["基频_Hz", "声压_dB_SPL", "共振峰_Hz"],
    "声道分区":     ["声带", "咽腔", "鼻腔", "口腔", "唇", "舌", "齿", "下颌"],
    "基频范围":     {"min": 60,   "max": 500,  "单位": "Hz",
                     "注释": "成年男声典型85-180Hz，女声165-255Hz"},
    "声压范围":     {"min": 60,   "max": 90,   "单位": "dB SPL",
                     "注释": "日常对话~60-70dB，喊叫~80-90dB"},
    "共振峰范围":   {"F1": {"min": 250,  "max": 1000, "单位": "Hz"},
                     "F2": {"min": 700,  "max": 2500, "单位": "Hz"},
                     "F3": {"min": 1500, "max": 3500, "单位": "Hz"}},
    "分辨率":       {"频率": "1–2 Hz", "强度": "0.5–1 dB"},
    "采样频率":     {"min": 16000, "max": 48000, "单位": "Hz"},
    "部位":         "8 声道分区（声带/咽腔/鼻腔/口腔/唇/舌/齿/下颌）",
    "状态":         "[判读·待核]"
}

详细分辨率 = {
    "基频_Hz":    {"频率分辨率": "1–2 Hz"},
    "声压_dB_SPL": {"强度分辨率": "0.5–1 dB"},
    "共振峰_Hz":   {"F1": "250-1000 Hz", "F2": "700-2500 Hz", "F3": "1500-3500 Hz"},
}

默认行 = 8     # 声道分区数
默认列 = 64    # 时间帧数


def _映射到范围(值, 入最小, 入最大, 出最小, 出最大):
    if 入最大 == 入最小:
        return (出最小 + 出最大) / 2.0
    规一 = (值 - 入最小) / (入最大 - 入最小)
    return 出最小 + 规一 * (出最大 - 出最小)


def _重采网格(帧二维, 行数, 列数):
    if not 帧二维 or not 帧二维[0]:
        return [[0.0] * 列数 for _ in range(行数)]
    源行数 = len(帧二维)
    源列数 = max(len(r) for r in 帧二维)
    输出 = []
    for i in range(行数):
        源_i = min(int(i * 源行数 / 行数), 源行数 - 1)
        行 = []
        for j in range(列数):
            源_j = min(int(j * 源列数 / 列数), len(帧二维[源_i]) - 1)
            行.append(float(帧二维[源_i][源_j]))
        输出.append(行)
    return 输出


def 帧转基频(帧二维, 行数=默认行, 列数=默认列):
    """映射到基频 (Hz)。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(60 + 500) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 60, 500) for v in r] for r in 输出]


def 帧转声压(帧二维, 行数=默认行, 列数=默认列):
    """映射到声压 (dB SPL)。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(60 + 90) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 60, 90) for v in r] for r in 输出]


def 帧转共振峰(帧二维, 行数=默认行, 列数=默认列):
    """映射到 F1 共振峰 (Hz)。用 F1 做代表，F2/F3 公式用 F1×固定比值。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(250 + 1000) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 250, 1000) for v in r] for r in 输出]


def 建测试帧(花样="gradient", 行数=默认行, 列数=默认列):
    """生成测试帧。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[(i + j) / (行数 + 列数) * 100.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 100) for j in range(列数)] for i in range(行数)]
    elif 花样 == "speech_footprint":
        # 声带活跃（基频+声压高），唇舌次级（共振峰调整），下颌辅助
        基线 = [150.0, 100.0, 60.0, 80.0, 120.0, 130.0, 50.0, 90.0]
        return [[基线[i] + j * 0.1 for j in range(列数)] for i in range(行数)]
    else:
        raise ValueError(f"未知花样: {花样}")


class 发声矩阵:
    """发声矩阵 — 8（声道分区）× 64（时间帧）。

    注意：三个子模态共用同一个尺寸和矩阵类，因为基频/声压/共振峰
    都是同一声学信号的不同特征，来自同一传感器（麦克风）。
    """

    子工厂 = {
        "基频_Hz":    帧转基频,
        "声压_dB_SPL":帧转声压,
        "共振峰_Hz":  帧转共振峰,
    }

    单位 = {
        "基频_Hz":     "Hz",
        "声压_dB_SPL": "dB SPL",
        "共振峰_Hz":   "Hz",
    }

    量程 = {
        "基频_Hz":     规格["基频范围"],
        "声压_dB_SPL": 规格["声压范围"],
        "共振峰_Hz":   规格["共振峰范围"],
    }

    def __init__(self, data, sub_modality="基频_Hz", source="test", ts=None):
        if sub_modality not in self.子工厂:
            raise ValueError(f"未知子模态: {sub_modality}")
        if len(data) != 默认行 or any(len(r) != 默认列 for r in data):
            raise ValueError(f"矩阵尺寸须为 {默认行}×{默认列}")
        self.data = data
        self.sub_modality = sub_modality
        self.source = source
        self.ts = ts or __import__('datetime').datetime.now().isoformat(timespec="milliseconds")

    def to_dict(self):
        return {
            "感官":       "发声",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8 声道分区 × 64 时间帧；测试尺寸",
            "取值范围":   self.量程[self.sub_modality],
            "分辨率":     详细分辨率[self.sub_modality],
            "采样频率":   规格["采样频率"],
            "部位":       规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]",
            "矩阵":       self.data,
        }

    def to_summary(self):
        return {
            "感官":       "发声",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8 声道分区 × 64 时间帧；测试尺寸",
            "取值范围":   self.量程[self.sub_modality],
            "分辨率":     详细分辨率[self.sub_modality],
            "采样频率":   规格["采样频率"],
            "部位":       规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]",
        }

    @staticmethod
    def from_frame(帧二维, sub_modality="基频_Hz", source="麦克风"):
        工厂 = 发声矩阵.子工厂[sub_modality]
        data = 工厂(帧二维)
        return 发声矩阵(data, sub_modality=sub_modality, source=source)


def 捕获发声(user, sub_modality="基频_Hz", source="麦克风", frame=None):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(花样="speech_footprint")
        source = f"test_speech_{sub_modality}"

    matrix = 发声矩阵.from_frame(frame, sub_modality=sub_modality, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    rec = t.append("发声帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    print(f"[发声矩阵] 尺寸 = {默认行}×{默认列}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  单位      = {发声矩阵.单位[m.sub_modality]}")
    print(f"  声道分区  = {'/'.join(规格['声道分区'])}")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.2f}, {max(flat):.2f}]")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=发声帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 言语声学/语音产生 [判读·待核]")
