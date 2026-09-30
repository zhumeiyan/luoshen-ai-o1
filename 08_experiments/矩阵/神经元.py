"""人类神经元信息矩阵 v0.1 — 帧→矩阵→入痕（待核）。

参考范围：皮层和海马等中枢神经系统的典型值。不是"洛神内部结构"，
而是"生物学锚点"——设计洛神数据结构的时间/空间尺度时的对照系。

子模态（五种数值化神经生理量）：
  放电频率_Hz     — 动作电位发放率（单个神经元或局部场电位）
  整合窗口_ms     — 突触整合时间常数（EPSP/IPSP 衰减窗口）
  振荡频段_Hz     — 频段边界范围 (delta/theta/alpha/beta/gamma)
  传导速度_m_s    — 轴突传导速度
  突触时延_ms     — 化学突触的突触小泡释放+受体+再充盈时间

矩阵结构：8 行（脑区：额叶/顶叶/颞叶/枕叶/海马/丘脑/小脑/基底节）
× 64 列（神经元采样点或特征维度）。

纪律：
  - 不做神经编码/解码/解释
  - 只做"参考量→物理量级→入痕"
  - 仅作为洛神动力学的时间尺度参考，不是等价物
  - 待升 [事实] 须神经科校验

实测说明：测试尺寸，设计值按脑区具体标定。

引文：
  - 放电频率：StatPearls. \"Neural Physiology - Electrophysiology.\" PMID:30725750
  - 整合窗口：StatPearls. \"Physiology, Synaptic Transmission.\" PMID:30844192
  - 传导速度：StatPearls. \"Physiology, Nerve Conduction.\" PMID:30844196
  - 突触时延：StatPearls. \"Physiology, Synaptic Transmission.\" PMID:30844192
  - 振荡频段：StatPearls. \"EEG Frequency Bands - StatPearls.\" PMID 30969642
  来源均取自 StatPearls Publishing LLC, Treasure Island (FL), 2026.
"""

# ── 神经生理参考量级（StatPearls + 教科书范围） ──
规格 = {
    "感官":         "人类神经元",
    "子模态":       ["放电频率_Hz", "整合窗口_ms", "振荡频段_Hz", "传导速度_m_s", "突触时延_ms"],
    "脑区":         ["额叶", "顶叶", "颞叶", "枕叶", "海马", "丘脑", "小脑", "基底节"],
    "放电频率范围": {"min": 0.1,  "max": 1000, "单位": "Hz",
                     "注释": "多数皮层神经元 <10 Hz；burst 状态可达 200-1000 Hz",
                     "引文": "StatPearls. \"Neural Physiology - Electrophysiology.\" PMID:30725750"},
    "整合窗口范围": {"min": 1,    "max": 200,   "单位": "ms",
                     "注释": "AMPA ~2-5 ms，NMDA ~50-200 ms，GABA_B ~50-200 ms",
                     "引文": "StatPearls. \"Physiology, Synaptic Transmission.\" PMID:30844192"},
    "振荡频段范围": {"min": 0.1,  "max": 300,   "单位": "Hz",
                     "注释": "delta 0.5-4，theta 4-8，alpha 8-13，beta 13-30，gamma 30-300",
                     "引文": "StatPearls. \"EEG Frequency Bands.\" PMID:30969642"},
    "传导速度范围": {"min": 0.5,  "max": 120,   "单位": "m/s",
                     "注释": "C纤维 ~0.5-2 m/s；Aα运动神经 ~70-120 m/s；皮层联合纤维 ~3-30 m/s",
                     "引文": "StatPearls. \"Physiology, Nerve Conduction.\" PMID:30844196"},
    "突触时延范围": {"min": 0.3,  "max": 3.0,   "单位": "ms",
                     "注释": "电突触 <0.1 ms；化学突触 ~0.3-3 ms（小泡融合+受体激活）",
                     "引文": "StatPearls. \"Physiology, Synaptic Transmission.\" PMID:30844192"},
    "部位":         "8 脑区（额/顶/颞/枕/海马/丘脑/小脑/基底节）",
    "状态":         "[判读·待核]"
}

详细分辨率 = {
    "放电频率_Hz":  {"步长": "~0.1 Hz (低频频段)"},
    "整合窗口_ms":  {"步长": "~0.1 ms (快突触)"},
    "振荡频段_Hz":  {"步长": "频段内连续, 按研究设定"},
    "传导速度_m_s": {"步长": "~0.5 m/s (类别差异)"},
    "突触时延_ms":  {"步长": "~0.1 ms (化学突触细节)"},
}

默认行 = 8
默认列 = 64


def _映射到范围(值, 入最小, 入最大, 出最小, 出最大):
    """线性映射到目标量程。"""
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


def 帧转放电频率(帧二维, 行数=默认行, 列数=默认列):
    """映射到放电频率 (Hz)。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(0.1 + 1000) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 0.1, 1000) for v in r] for r in 输出]


def 帧转整合窗口(帧二维, 行数=默认行, 列数=默认列):
    """映射到整合窗口 (ms)。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(1 + 200) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 1, 200) for v in r] for r in 输出]


def 帧转振荡频段(帧二维, 行数=默认行, 列数=默认列):
    """映射到振荡频段 (Hz)。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(0.1 + 300) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 0.1, 300) for v in r] for r in 输出]


def 帧转传导速度(帧二维, 行数=默认行, 列数=默认列):
    """映射到传导速度 (m/s)。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(0.5 + 120) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 0.5, 120) for v in r] for r in 输出]


def 帧转突触时延(帧二维, 行数=默认行, 列数=默认列):
    """映射到突触时延 (ms)。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(0.3 + 3.0) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 0.3, 3.0) for v in r] for r in 输出]


def 建测试帧(花样="gradient", 行数=默认行, 列数=默认列):
    """生成测试帧。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[(i + j) / (行数 + 列数) * 100.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 100) for j in range(列数)] for i in range(行数)]
    elif 花样 == "neural_footprint":
        # 额叶/海马/基底节活跃(高发放)，丘脑/小脑中等，顶/颞/枕相对静止
        基线 = [80.0, 30.0, 25.0, 20.0, 70.0, 45.0, 50.0, 65.0]
        return [[基线[i] + j * 0.05 for j in range(列数)] for i in range(行数)]
    else:
        raise ValueError(f"未知花样: {花样}")


class 神经元矩阵:
    """人类神经元信息矩阵 — 8（脑区）× 64（神经元采样/维度）。"""

    子工厂 = {
        "放电频率_Hz":  帧转放电频率,
        "整合窗口_ms":  帧转整合窗口,
        "振荡频段_Hz":  帧转振荡频段,
        "传导速度_m_s": 帧转传导速度,
        "突触时延_ms": 帧转突触时延,
    }

    单位 = {
        "放电频率_Hz":  "Hz",
        "整合窗口_ms":  "ms",
        "振荡频段_Hz":  "Hz",
        "传导速度_m_s": "m/s",
        "突触时延_ms": "ms",
    }

    量程 = {
        "放电频率_Hz":  规格["放电频率范围"],
        "整合窗口_ms":  规格["整合窗口范围"],
        "振荡频段_Hz":  规格["振荡频段范围"],
        "传导速度_m_s": 规格["传导速度范围"],
        "突触时延_ms": 规格["突触时延范围"],
    }

    def __init__(self, data, sub_modality="放电频率_Hz", source="test", ts=None):
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
            "感官":       "人类神经元",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8 脑区 × 64 神经采样；测试尺寸",
            "取值范围":   self.量程[self.sub_modality],
            "分辨率":     详细分辨率[self.sub_modality],
            "采样频率":   "— (静态参考量，非连续采样)",
            "部位":       规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]",
            "矩阵":       self.data,
        }

    def to_summary(self):
        return {
            "感官":       "人类神经元",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8 脑区 × 64 神经采样；测试尺寸",
            "取值范围":   self.量程[self.sub_modality],
            "分辨率":     详细分辨率[self.sub_modality],
            "采样频率":   "— (静态参考量，非连续采样)",
            "部位":       规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]",
        }

    @staticmethod
    def from_frame(帧二维, sub_modality="放电频率_Hz", source="神经生理参考"):
        工厂 = 神经元矩阵.子工厂[sub_modality]
        data = 工厂(帧二维)
        return 神经元矩阵(data, sub_modality=sub_modality, source=source)


def 捕获神经元(user, sub_modality="放电频率_Hz", source="神经生理参考", frame=None):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(花样="neural_footprint")
        source = f"test_{sub_modality}"

    matrix = 神经元矩阵.from_frame(frame, sub_modality=sub_modality, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    rec = t.append("神经元参考帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    print(f"[神经元矩阵] 尺寸 = {默认行}×{默认列}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  单位      = {神经元矩阵.单位[m.sub_modality]}")
    print(f"  脑区      = {'/'.join(规格['脑区'])}")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.2f}, {max(flat):.2f}]")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=神经元参考帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 神经生理学教科书范围 [判读·待核]")
