"""温度觉矩阵 v0.1 — 帧→矩阵→入痕（待核）。

规格来源：表A·v0.4·温度觉行。

两子模态：
  - 冷觉_°C：0–35 °C，CDT（冷检测阈）手 −2.2~−3.6 °C，足 −2.6~−4.5 °C
  - 热觉_°C：35–50 °C（无痛上限），WDT [槽·待补]

分辨率：敏感区 0.02–0.1 °C
采样频率：1–10 Hz
部位：8 行（身体八区：指尖/掌/躯干/前臂/上臂/背/腿/足）

纪律：
  - 不做温觉识别/分类/推理
  - 只做"帧→摄氏温标映射→入痕"
  - 冷/热分界 35 °C（皮肤温度舒适区上界）
  - 待升 [事实] 需哨兵核

矩阵结构：8 行（身体分区）× 64 列（测量点）。
实测说明：测试尺寸，设计值按部位具体标定。
"""

# ── 物理规格（表A·v0.4·温度觉行） ──
规格 = {
    "感官":       "温度觉",
    "子模态":     ["冷觉_°C", "热觉_°C"],
    "身体分区":   ["指尖", "掌", "前臂", "上臂", "背", "躯干", "腿", "足"],
    "冷觉范围":   {"min": 0,   "max": 35,  "单位": "°C",
                   "CDT手": "−2.2~−3.6 °C（相对基线32 °C）",
                   "CDT足": "−2.6~−4.5 °C（相对基线32 °C）"},
    "热觉范围":   {"min": 35,  "max": 50,  "单位": "°C",
                   "WDT": "[槽·待补]"},
    "分辨率":     {"冷觉_敏感区": "0.02–0.1 °C",
                   "热觉_敏感区": "0.02–0.1 °C"},
    "采样频率":   {"min": 1, "max": 10, "单位": "Hz"},
    "部位":       "8 分区（指尖/掌/前臂/上臂/背/躯干/腿/足）",
    "状态":       "[判读·待核]"
}

详细分辨率 = {
    "冷觉_°C": {"冷觉_敏感区": "0.02–0.1 °C"},
    "热觉_°C": {"热觉_敏感区": "0.02–0.1 °C"},
}

默认行 = 8      # 身体分区数
默认列 = 64     # 每区测量点数


def 帧转摄氏(帧二维, 行数=默认行, 列数=默认列, 模式="cold"):
    """把任意帧映射到摄氏温标。

    模式="cold" → 0–35 °C 范围
    模式="hot"  → 35–50 °C 范围
    """
    if not 帧二维 or not 帧二维[0]:
        raise ValueError("帧二维 为空")

    源行数 = len(帧二维)
    源列数 = max(len(r) for r in 帧二维)

    输出 = []
    for i in range(行数):
        源_i = min(int(i * 源行数 / 行数), 源行数 - 1)
        r = []
        for j in range(列数):
            源_j = min(int(j * 源列数 / 列数), len(帧二维[源_i]) - 1)
            r.append(float(帧二维[源_i][源_j]))
        输出.append(r)

    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)

    if 模式 == "cold":
        目标最小 = 规格["冷觉范围"]["min"]
        目标最大 = 规格["冷觉范围"]["max"]
    elif 模式 == "hot":
        目标最小 = 规格["热觉范围"]["min"]
        目标最大 = 规格["热觉范围"]["max"]
    else:
        raise ValueError(f"未知模式: {模式}")

    if 原最大 == 原最小:
        return [[(目标最小 + 目标最大) / 2] * 列数 for _ in range(行数)]

    映射后 = []
    for r in 输出:
        行 = [_映射到范围(v, 原最小, 原最大, 目标最小, 目标最大) for v in r]
        映射后.append(行)
    return 映射后


def _映射到范围(值, 入最小, 入最大, 出最小, 出最大):
    if 入最大 == 入最小:
        return (出最小 + 出最大) / 2.0
    规一 = (值 - 入最小) / (入最大 - 入最小)
    return 出最小 + 规一 * (出最大 - 出最小)


def 建测试帧(花样="gradient", 行数=默认行, 列数=默认列):
    """生成测试帧。花样: gradient / random / cold_footprint / hot_footprint。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[(i + j) / (行数 + 列数) * 50.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 50) for j in range(列数)] for i in range(行数)]
    elif 花样 == "cold_footprint":
        # 手区偏冷(指尖/掌)，躯干/腿接近基线
        基线 = [5.0, 8.0, 20.0, 22.0, 24.0, 25.0, 28.0, 30.0]  # °C
        return [[基线[i] + j * 0.02 for j in range(列数)] for i in range(行数)]
    elif 花样 == "hot_footprint":
        # 手区偏热，躯干/腿接近基线
        基线 = [42.0, 40.0, 38.0, 37.0, 36.5, 36.0, 35.5, 35.0]  # °C
        return [[基线[i] + (列数 - j) * 0.02 for j in range(列数)] for i in range(行数)]
    else:
        raise ValueError(f"未知花样: {花样}")


class 温矩阵:
    """温度觉矩阵 — 8（身体分区）× 64（测量点）。"""

    单位 = {
        "冷觉_°C": "°C",
        "热觉_°C": "°C",
    }

    量程 = {
        "冷觉_°C": 规格["冷觉范围"],
        "热觉_°C": 规格["热觉范围"],
    }

    产出模式 = {
        "冷觉_°C": "cold",
        "热觉_°C": "hot",
    }

    def __init__(self, data, sub_modality="冷觉_°C", source="test", ts=None):
        if sub_modality not in self.量程:
            raise ValueError(f"未知子模态: {sub_modality}")
        if len(data) != 默认行 or any(len(r) != 默认列 for r in data):
            raise ValueError(f"矩阵尺寸须为 {默认行}×{默认列}")
        self.data = data
        self.sub_modality = sub_modality
        self.source = source
        self.ts = ts or __import__('datetime').datetime.now().isoformat(timespec="milliseconds")

    def to_dict(self):
        return {
            "感官":       "温度觉",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8 分区 × 64 测量点；测试尺寸",
            "取值范围":   self.量程[self.sub_modality],
            "分辨率":   详细分辨率[self.sub_modality],
            "采样频率": 规格["采样频率"],
            "部位":     规格["部位"],
            "来源":     self.source,
            "时间戳":   self.ts,
            "状态":     "[判读·待核]",
            "矩阵":     self.data,
        }

    def to_summary(self):
        return {
            "感官":     "温度觉",
            "子模态":   self.sub_modality,
            "尺寸":     f"{默认行}×{默认列}",
            "尺寸说明": "8 分区 × 64 测量点；测试尺寸",
            "取值范围": self.量程[self.sub_modality],
            "分辨率":  详细分辨率[self.sub_modality],
            "采样频率":  规格["采样频率"],
            "部位":      规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]",
        }

    @staticmethod
    def from_frame(帧二维, sub_modality="冷觉_°C", source="温度传感器"):
        模式 = 温矩阵.产出模式[sub_modality]
        data = 帧转摄氏(帧二维, 模式=模式)
        return 温矩阵(data, sub_modality=sub_modality, source=source)


def 捕获温度(user, sub_modality="冷觉_°C", source="温度传感器", frame=None):
    """帧→矩阵→入痕主入口。"""
    if sub_modality == "冷觉_°C":
        默认花样 = "cold_footprint"
    elif sub_modality == "热觉_°C":
        默认花样 = "hot_footprint"
    else:
        默认花样 = "gradient"

    if frame is None:
        frame = 建测试帧(花样=默认花样)
        source = f"test_{默认花样}_{sub_modality}"

    matrix = 温矩阵.from_frame(frame, sub_modality=sub_modality, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    rec = t.append("温度觉帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    print(f"[温度觉矩阵] 尺寸 = {默认行}×{默认列}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  单位      = {温矩阵.单位[m.sub_modality]}")
    print(f"  分区      = {'/'.join(规格['身体分区'])}")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.2f}, {max(flat):.2f}] °C")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=温度觉帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 表A·v0.4·温度觉行 [判读·待核]")
