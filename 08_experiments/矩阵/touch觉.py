"""触觉矩阵 v0.1 — 帧→矩阵→入痕（待核）。

规格来源：表A·v0.4·触觉行。

三子模态：
  - 压力_kPa：0.01–100 kPa，两点分辨阈 2 mm(指尖)→40(背)，≥100 Hz
  - 振动_um：0.1–1000 Hz，阈值 ~0.01 μm，≥1000 Hz
  - 纹理_Ra_um：0.025–100 μm，纹理分辨 ~0.1 mm

纪律：
  - 不做触觉识别/分类/推理
  - 只做"测量阵列→物理量映射→入痕"
  - 待升 [事实] 需哨兵核

矩阵结构：8 行（身体分区：指尖/掌/前臂/上臂/背/躯干/腿/足）× 64 列（测量点/时间帧）。
实测说明：测试尺寸，设计值按部位具体标定。
"""

# ── 物理规格（表A·v0.4·触觉行） ──
规格 = {
    "感官":       "触觉",
    "子模态":     ["压力_kPa", "振动_um", "纹理_Ra_um"],
    "身体分区":   ["指尖", "掌", "前臂", "上臂", "背", "躯干", "腿", "足"],
    "压力范围":   {"min": 0.01, "max": 100, "单位": "kPa"},
    "振动范围":   {"min": 0.1,  "max": 1000, "单位": "Hz"},
    "纹理范围":   {"min": 0.025, "max": 100, "单位": "μm Ra"},
    "分辨率":     {"两点分辨": "2 mm(指尖)→40 mm(背)",
                   "振动阈值": "~0.01 μm",
                   "纹理分辨": "~0.1 mm"},
    "采样频率":   {"压力_min": 100, "振动_min": 1000, "纹理_min": 0, "单位": "Hz"},
    "部位":       "8 分区（指尖/掌/前臂/上臂/背/躯干/腿/足）",
    "状态":       "[判读·待核]"
}

详细分辨率 = {
    "压力_kPa":   {"两点分辨": "2 mm(指尖)→40 mm(背)"},
    "振动_um":    {"振动阈值": "~0.01 μm"},
    "纹理_Ra_um": {"纹理分辨": "~0.1 mm"},
}

默认行 = 8      # 身体分区数
默认列 = 64     # 每区测量点数


def _映射到范围(值, 入最小, 入最大, 出最小, 出最大):
    """线性映射到目标量程。"""
    if 入最大 == 入最小:
        return (出最小 + 出最大) / 2.0
    规一 = (值 - 入最小) / (入最大 - 入最小)
    return 出最小 + 规一 * (出最大 - 出最小)


def 帧转压力(帧二维, 行数=默认行, 列数=默认列):
    """映射到 kPa。输入为任意数值阵。"""
    return _映射帧(帧二维, 行数, 列数, 规格["压力范围"]["min"], 规格["压力范围"]["max"])


def 帧转振动(帧二维, 行数=默认行, 列数=默认列):
    """映射到 μm（振动位移幅值）。输入为任意数值阵。"""
    import math
    输出 = _重采网格(帧二维, 行数, 列数)
    # 振动位移对数映射：0.01 μm ~ 100 μm
    振最小, 振最大 = 0.01, 100.0
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        中 = math.sqrt(振最小 * 振最大)
        return [[中] * 列数 for _ in range(行数)]
    映射后 = []
    for r in 输出:
        行 = []
        for v in r:
            规一 = (v - 原最小) / (原最大 - 原最小)
            对数值 = 振最小 * math.pow(振最大 / 振最小, 规一)
            行.append(对数值)
        映射后.append(行)
    return 映射后


def 帧转纹理(帧二维, 行数=默认行, 列数=默认列):
    """映射到 μm Ra（粗糙度）。输入为任意数值阵。"""
    return _映射帧(帧二维, 行数, 列数, 规格["纹理范围"]["min"], 规格["纹理范围"]["max"])


def _映射帧(帧二维, 行数, 列数, 目标最小, 目标最大):
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(目标最小 + 目标最大) / 2.0] * 列数 for _ in range(行数)]
    映射后 = []
    for r in 输出:
        行 = [_映射到范围(v, 原最小, 原最大, 目标最小, 目标最大) for v in r]
        映射后.append(行)
    return 映射后


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


def 建测试帧(花样="gradient", 行数=默认行, 列数=默认列):
    """生成测试帧。花样: gradient / random / press_pattern。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[(i + j) / (行数 + 列数) * 50.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 50) for j in range(列数)] for i in range(行数)]
    elif 花样 == "press_pattern":
        # 模拟指尖高压、掌中压、其余低压
        基线 = [80.0, 50.0, 20.0, 15.0, 10.0, 8.0, 5.0, 3.0]
        return [[基线[i] + j * 0.1 for j in range(列数)] for i in range(行数)]
    else:
        raise ValueError(f"未知花样: {花样}")


class 触矩阵:
    """触觉矩阵 — 8（身体分区）× 64（测量点）。"""

    子工厂 = {
        "压力_kPa":  帧转压力,
        "振动_um":   帧转振动,
        "纹理_Ra_um": 帧转纹理,
    }

    单位 = {
        "压力_kPa":  "kPa",
        "振动_um":   "μm",
        "纹理_Ra_um": "μm Ra",
    }

    量程 = {
        "压力_kPa":  规格["压力范围"],
        "振动_um":   {"min": 0.01, "max": 100, "单位": "μm"},
        "纹理_Ra_um": 规格["纹理范围"],
    }

    def __init__(self, data, sub_modality="压力_kPa", source="test", ts=None):
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
            "感官":       "触觉",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8 分区（指尖→足）× 64 测量点；测试尺寸",
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
        """紧凑版入痕用（不存原始矩阵）。"""
        return {
            "感官":       "触觉",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明": "8 分区 × 64 测量点",
            "取值范围": self.量程[self.sub_modality],
            "分辨率":    详细分辨率[self.sub_modality],
            "采样频率":  规格["采样频率"],
            "部位":      规格["部位"],
            "来源":      self.source,
            "时间戳":    self.ts,
            "状态":      "[判读·待核]",
        }

    @staticmethod
    def from_frame(帧二维, sub_modality="压力_kPa", source="传感器阵列"):
        工厂 = 触矩阵.子工厂[sub_modality]
        data = 工厂(帧二维)
        return 触矩阵(data, sub_modality=sub_modality, source=source)


def 捕获触觉(user, sub_modality="压力_kPa", source="传感器阵列", frame=None):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(花样="gradient")
        source = f"test_gradient_{sub_modality}"

    matrix = 触矩阵.from_frame(frame, sub_modality=sub_modality, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    rec = t.append("触觉帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    print(f"[触觉矩阵] 尺寸 = {默认行}×{默认列}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  单位      = {触矩阵.单位[m.sub_modality]}")
    print(f"  分区      = {'/'.join(规格['身体分区'])}")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.4f}, {max(flat):.4f}]")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=触觉帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 表A·v0.4·触觉行 [判读·待核]")
