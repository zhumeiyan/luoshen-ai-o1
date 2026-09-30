"""痛觉矩阵 v0.1 — 帧→矩阵→入痕（待核）。

规格来源：表A·v0.4·痛觉行。

五子模态（疼痛医学阈值）：
  - 机械痛_kPa：50–500 kPa（PPT 压痛阈值），分辨率 ~10 kPa
  - 热痛_°C：43–49 °C（HPT 热痛阈值），分辨率 ~0.5 °C
  - 冷痛_°C：0–10 °C（CPT 冷痛阈值），分辨率 ~1 °C
  - 电刺激_mA：0.1–10 mA，分辨率 ~0.1 mA
  - 化学痛_分：量表 0–10（VAS/NRS），分辨率 ~0.5 分

采样频率：—（阈值测量不连续，事件触发）
部位：8 行（身体分区：指尖/掌/前臂/上臂/背/躯干/腿/足）

纪律：
  - 不做痛觉识别/分类/推理
  - 只做"帧→阈值/量表映射→入痕"
  - 阈值含义：触发痛觉的最小强度
  - 待升 [事实] 需哨兵核

矩阵结构：8 行（身体分区）× 64 列（测量点）。
实测说明：测试尺寸，设计值按部位具体标定。
表A外补充：电刺激、化学痛，表A v0.4 未列，本文件保留。
"""

# ── 物理规格（表A·v0.4·痛觉行） ──
规格 = {
    "感官":       "痛觉",
    "子模态":     ["机械痛_kPa", "热痛_°C", "冷痛_°C", "电刺激_mA", "化学痛_分"],
    "身体分区":   ["指尖", "掌", "前臂", "上臂", "背", "躯干", "腿", "足"],
    "机械痛范围": {"min": 50,   "max": 500, "单位": "kPa", "简称": "PPT"},
    "热痛范围":   {"min": 43,   "max": 49,  "单位": "°C",  "简称": "HPT"},
    "冷痛范围":   {"min": 0,    "max": 10,  "单位": "°C",  "简称": "CPT"},
    "电刺激范围": {"min": 0.1,  "max": 10,  "单位": "mA"},
    "化学痛范围": {"min": 0,    "max": 10,  "单位": "分",  "简称": "VAS/NRS"},
    "分辨率":     {"机械痛": "~10 kPa", "热痛": "~0.5 °C", "冷痛": "~1 °C",
                   "电刺激": "~0.1 mA", "化学痛": "~0.5 分"},
    "采样频率":   {"说明": "阈值测量为事件触发，不连续采样"},
    "部位":       "8 分区（指尖/掌/前臂/上臂/背/躯干/腿/足）",
    "状态":       "[判读·待核]"
}

详细分辨率 = {
    "机械痛_kPa":  {"机械痛": "~10 kPa"},
    "热痛_°C":     {"热痛": "~0.5 °C"},
    "冷痛_°C":     {"冷痛": "~1 °C"},
    "电刺激_mA":   {"电刺激": "~0.1 mA"},
    "化学痛_分":   {"化学痛": "~0.5 分"},
}

默认行 = 8
默认列 = 64


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


def 帧转机械痛(帧二维, 行数=默认行, 列数=默认列):
    """映射到 kPa（压痛阈值 50–500）。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    return [[_映射到范围(v, 原最小, 原最大, 50, 500) for v in r] for r in 输出]


def 帧转热痛(帧二维, 行数=默认行, 列数=默认列):
    """映射到 °C（热痛阈值 43–49）。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    return [[_映射到范围(v, 原最小, 原最大, 43, 49) for v in r] for r in 输出]


def 帧转冷痛(帧二维, 行数=默认行, 列数=默认列):
    """映射到 °C（冷痛 0–10）。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    return [[_映射到范围(v, 原最小, 原最大, 0, 10) for v in r] for r in 输出]


def 帧转电刺激(帧二维, 行数=默认行, 列数=默认列):
    """映射到 mA（电刺激阈值 0.1–10）。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    return [[_映射到范围(v, 原最小, 原最大, 0.1, 10) for v in r] for r in 输出]


def 帧转化学痛(帧二维, 行数=默认行, 列数=默认列):
    """映射到 分（VAS/NRS 量表 0–10）。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    return [[round(_映射到范围(v, 原最小, 原最大, 0, 10), 1) for v in r] for r in 输出]


def 建测试帧(花样="gradient", 行数=默认行, 列数=默认列):
    """生成测试帧。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[(i + j) / (行数 + 列数) * 100.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 100) for j in range(列数)] for i in range(行数)]
    elif 花样 == "pain_threshold_footprint":
        # 阈值分布：指尖<掌<近端（敏感→不敏感）
        基线 = [80.0, 120.0, 200.0, 250.0, 300.0, 320.0, 350.0, 400.0]
        return [[基线[i] + j * 0.2 for j in range(列数)] for i in range(行数)]
    else:
        raise ValueError(f"未知花样: {花样}")


class 痛矩阵:
    """痛觉矩阵 — 8（身体分区）× 64（测量点）。"""

    子工厂 = {
        "机械痛_kPa":  帧转机械痛,
        "热痛_°C":     帧转热痛,
        "冷痛_°C":     帧转冷痛,
        "电刺激_mA":   帧转电刺激,
        "化学痛_分":   帧转化学痛,
    }

    单位 = {
        "机械痛_kPa":  "kPa",
        "热痛_°C":     "°C",
        "冷痛_°C":     "°C",
        "电刺激_mA":   "mA",
        "化学痛_分":   "分",
    }

    量程 = {
        "机械痛_kPa":  规格["机械痛范围"],
        "热痛_°C":     规格["热痛范围"],
        "冷痛_°C":     规格["冷痛范围"],
        "电刺激_mA":   规格["电刺激范围"],
        "化学痛_分":   规格["化学痛范围"],
    }

    def __init__(self, data, sub_modality="机械痛_kPa", source="test", ts=None):
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
            "感官":       "痛觉",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8 分区 × 64 测量点；测试尺寸",
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
            "感官":       "痛觉",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8 分区 × 64 测量点；测试尺寸",
            "取值范围":   self.量程[self.sub_modality],
            "分辨率":     详细分辨率[self.sub_modality],
            "采样频率":   规格["采样频率"],
            "部位":       规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]",
        }

    @staticmethod
    def from_frame(帧二维, sub_modality="机械痛_kPa", source="疼痛阈值仪"):
        工厂 = 痛矩阵.子工厂[sub_modality]
        data = 工厂(帧二维)
        return 痛矩阵(data, sub_modality=sub_modality, source=source)


def 捕获痛觉(user, sub_modality="机械痛_kPa", source="疼痛阈值仪", frame=None):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(花样="pain_threshold_footprint")
        source = f"test_threshold_{sub_modality}"

    matrix = 痛矩阵.from_frame(frame, sub_modality=sub_modality, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    rec = t.append("痛觉帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    print(f"[痛觉矩阵] 尺寸 = {默认行}×{默认列}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  单位      = {痛矩阵.单位[m.sub_modality]}")
    print(f"  分区      = {'/'.join(规格['身体分区'])}")
    print(f"  含义      = 痛觉阈值（触发痛的最小强度）")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.2f}, {max(flat):.2f}]")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=痛觉帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 表A·v0.4·痛觉行 [判读·待核]")
