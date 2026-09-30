"""听觉矩阵 v0.1 — 帧→矩阵→入痕（终版）。

规格来源：表A·v0.4·听觉行。

纪律：
  - 不做过信号处理（无频谱分析、无特征提取）
  - 只做"帧→dB SPL映射→入痕"
  - 终版闭合

实测说明：
  - 48×64 = 测试尺寸（测试值，设计值按频率bin已另算）
  - 当前数据源为合成测试信号（test_tone_sweep），不是麦克风真帧
  - 时间戳毫秒精度，避免同秒碰撞
  - 字段对齐视觉：13 项字段完全对应
"""

# ── 物理规格（表A·v0.4·听觉行） ──
规格 = {
    "设备型号":   "待填(暂无真实麦克风型号)",
    "感官":       "听觉",
    "子模态":     ["频率_Hz", "声压_dB_SPL"],
    "频率范围":   {"min": 20, "max": 20000, "单位": "Hz"},
    "声压范围":   {"min": 0, "max": 120, "单位": "dB SPL"},
    "分辨率":     {"频率_JND": "0.2–0.3%", "强度_JND": "0.5–1 dB"},
    "采样频率":   {"min": 44100, "单位": "Hz"},
    "部位":       "双耳",
    "状态":       "[事实·终版]"
}

默认频带数 = 64
默认帧数 = 48
默认采样率 = 44100


def 帧转分贝声压(帧二维, 行数=默认帧数, 列数=默认频带数):
    """把任意二维声压数组映射到 dB SPL。"""
    if not 帧二维 or not 帧二维[0]:
        raise ValueError("帧二维 为空")

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

    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    目标最小 = 规格["声压范围"]["min"]
    目标最大 = 规格["声压范围"]["max"]

    if 原最大 == 原最小:
        return [[(目标最大 + 目标最小) / 2] * 列数 for _ in range(行数)], {
            "映射方式": "常量填充（全等输入）"
        }

    映射后 = []
    for r in 输出:
        行 = [目标最小 + (v - 原最小) / (原最大 - 原最小) * (目标最大 - 目标最小)
               for v in r]
        映射后.append(行)

    return 映射后, {"映射方式": "线性拉伸→dB SPL"}


def 建测试帧(行数=默认帧数, 列数=默认频带数, 花样="tone_sweep"):
    """生成测试帧。花样: tone_sweep / white_noise / pure_tone / silence / loud。"""
    import random
    random.seed(42)
    if 花样 == "tone_sweep":
        return [[20 + (j / 列数) * 19980 for j in range(列数)] for i in range(行数)]
    elif 花样 == "white_noise":
        return [[random.uniform(20, 80) for j in range(列数)] for i in range(行数)]
    elif 花样 == "pure_tone":
        return [[1000.0 + i * 10] * 列数 for i in range(行数)]
    elif 花样 == "silence":
        return [[0.0 + i * 0.01] * 列数 for i in range(行数)]
    elif 花样 == "loud":
        return [[120.0 - i * 0.5] * 列数 for i in range(行数)]
    else:
        raise ValueError(f"未知花样: {花样}")


class 听矩阵:
    """听觉矩阵 — 48×64 声压网格（dB SPL）。"""

    def __init__(self, data, source="test", ts=None, sub_modality="声压_dB_SPL"):
        if len(data) != 默认帧数 or any(len(r) != 默认频带数 for r in data):
            raise ValueError(f"矩阵尺寸须为 {默认帧数}×{默认频带数}")
        self.data = data
        self.source = source
        self.sub_modality = sub_modality
        self.ts = ts or __import__('datetime').datetime.now().isoformat(timespec="milliseconds")

    def to_dict(self):
        return {
            "设备型号": 规格["设备型号"],
            "感官":       "听觉",
            "子模态":    规格["子模态"],
            "尺寸":       f"{默认帧数}×{默认频带数}",
            "尺寸说明":   "测试尺寸，按频率bin计算",
            "取值范围":   {"min_dB_SPL": 0, "max_dB_SPL": 120},
            "分辨率":     规格["分辨率"],
            "采样频率":   规格["采样频率"],
            "部位":       规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]",
            "矩阵":       self.data,
        }

    def to_summary(self):
        """紧凑版入痕用。"""
        return {
            "设备型号": 规格["设备型号"],
            "感官":       "听觉",
            "子模态":    规格["子模态"],
            "尺寸":       f"{默认帧数}×{默认频带数}",
            "尺寸说明":   "测试尺寸，按频率bin计算",
            "取值范围":   {"min_dB_SPL": 0, "max_dB_SPL": 120},
            "分辨率":     规格["分辨率"],
            "采样频率":   规格["采样频率"],
            "部位":       规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]"
        }

    @staticmethod
    def from_frame(帧二维, source="麦克风", sub_modality="声压_dB_SPL"):
        data, _ = 帧转分贝声压(帧二维)
        return 听矩阵(data, source=source, sub_modality=sub_modality)


def 捕获听觉(user, source="麦克风", frame=None):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(花样="tone_sweep")
        source = "test_tone_sweep"

    matrix = 听矩阵.from_frame(frame, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    rec = t.append("听觉帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    print(f"[听觉矩阵]   尺寸 = {默认帧数}×{默认频带数}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.1f}, {max(flat):.1f}] dB SPL  (设计范围 0–120)")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=听觉帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 表A·v0.4·听觉行 [待核]")
