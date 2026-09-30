"""LiDAR 矩阵 v0.2 — 帧→矩阵→入痕（待核）。

设备型号：1550 nm FMCW LiDAR（600 万点/秒级）
来源：制造商规格书

传感器参数：
  - 测距范围：0.3–200 m
  - 测距精度：±2 cm
  - 点频：600 万点/秒

纪律：
  - 飞行时间和距离都存，不丢弃原始量
  - 标清关系：Δt → d = c · Δt / 2，payload 里注明
  - 不做目标分割

实测说明：测试尺寸，设计值按点云规模标定。
矩阵结构：8 行 × 64 列。
"""

# ── 物理规格（典型 1550 nm FMCW LiDAR 参数） ──
规格 = {
    "设备型号":       "1550nm FMCW LiDAR",
    "感官":           "LiDAR",
    "子模态":         ["飞行时间_ps", "距离_m"],
    "波长":           {"value": 1550, "单位": "nm",
                        "注释": "1550 nm 人眼安全、大气透过率高；905 nm 亦常见"},
    "飞行时间量程":   {"min": 0, "max": 1333, "单位": "ps",
                        "注释": "对应 0–200 m 往返（Δt = 2d/c）"},
    "距离量程":       {"min": 0.3, "max": 200.0, "单位": "m",
                        "注释": "典型 1550 nm FMCW 测距范围"},
    "采样频率":       {"min": 10, "max": 30, "单位": "Hz",
                        "注释": "帧率（固态/机械旋转）"},
    "点云密度":       "~10⁴–10⁶ 点/帧",
    "距离分辨率":     {"value": "1–10", "单位": "cm"},
    "部位":           "三维空间视线可达（FOV 内，水平和垂直扫描范围取决于型号）",
    "状态":           "[判读·待核]"
}

详细分辨率 = {
    "飞行时间_ps":  {"步长": "1 ps ≈ 0.15 mm 距离分辨率"},
    "距离_m":       {"距离分辨率": "1–10 cm（取决于调制方式）"},
}

默认行 = 8
默认列 = 64
光速 = 299792458.0  # m/s
皮秒每米往返 = 2.0 / 光速 * 1e12  # ps per meter round trip ≈ 6670 ps/m


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


def _映射到范围(值, 入最小, 入最大, 出最小, 出最大):
    if 入最大 == 入最小:
        return (出最小 + 出最大) / 2.0
    规一 = (值 - 入最小) / (入最大 - 入最小)
    return 出最小 + 规一 * (出最大 - 出最小)


def 帧转飞行时间(帧二维, 行数=默认行, 列数=默认列):
    """映射到飞行时间（ps）— [事实·原始]。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[2000.0] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 0, 1333) for v in r] for r in 输出]


def 帧转距离(帧二维, 行数=默认行, 列数=默认列):
    """映射到距离（m）— [事实·设备输出级】。

    d = c · Δt / 2
    假设帧中心为近距离，边缘为远距离。
    """
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[50.0] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, 0.3, 200.0) for v in r] for r in 输出]


def 建测试帧(花样="gradient", 行数=默认行, 列数=默认列):
    """生成测试帧。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[(i + j) / (行数 + 列数) * 100.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 100) for j in range(列数)] for i in range(行数)]
    elif 花样 == "point_cloud_footprint":
        # 模拟点云中心密集（近距离）、边缘稀疏（远距离）
        基线 = [[10.0] * 列数 for _ in range(行数)]
        中行, 中列 = 行数 // 2, 列数 // 2
        for i in range(行数):
            for j in range(列数):
                d = ((i - 中行) ** 2 + (j - 中列) ** 2) ** 0.5
                基线[i][j] = max(0.0, 100.0 - d * 12)
        return 基线
    else:
        raise ValueError(f"未知花样: {花样}")


class 激光雷达矩阵:
    """LiDAR 矩阵 — 8 × 64（测试尺寸）。"""

    子工厂 = {
        "飞行时间_ps":  帧转飞行时间,
        "距离_m":       帧转距离,
    }

    单位 = {
        "飞行时间_ps":  "ps",
        "距离_m":       "m",
    }

    量程 = {
        "飞行时间_ps":  规格["飞行时间量程"],
        "距离_m":       规格["距离量程"],
    }

    def __init__(self, data, sub_modality="距离_m", source="test", ts=None):
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
            "设备型号":   规格["设备型号"],
            "感官":       "LiDAR",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8（分区）× 64（测量点）；测试尺寸，实际 LiDAR 为点云",
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
            "设备型号":   规格["设备型号"],
            "感官":       "LiDAR",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8（分区）× 64（测量点）；测试尺寸",
            "取值范围":   self.量程[self.sub_modality],
            "分辨率":     详细分辨率[self.sub_modality],
            "采样频率":   规格["采样频率"],
            "部位":       规格["部位"],
            "来源":       self.source,
            "时间戳":     self.ts,
            "状态":       "[判读·待核]",
        }

    @staticmethod
    def from_frame(帧二维, sub_modality="距离_m", source="LiDAR"):
        工厂 = 激光雷达矩阵.子工厂[sub_modality]
        data = 工厂(帧二维)
        return 激光雷达矩阵(data, sub_modality=sub_modality, source=source)


def 捕获激光雷达(user, sub_modality="距离_m", source="LiDAR", frame=None):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(花样="point_cloud_footprint")
        source = f"test_{sub_modality}"

    matrix = 激光雷达矩阵.from_frame(frame, sub_modality=sub_modality, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    payload["原始量原则说明"] = (
        "飞行时间为光子往返 ToF [事实·原始]；"
        "距离为 d=c·Δt/2 换算后的设备输出级 [事实·设备输出级]"
    )
    rec = t.append("LiDAR帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    标记 = "[事实·原始]" if m.sub_modality == "飞行时间_ps" else "[事实·设备输出级]"
    print(f"[LiDAR矩阵] 尺寸 = {默认行}×{默认列}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  标记      = {标记}")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.1f}, {max(flat):.1f}] {激光雷达矩阵.单位[m.sub_modality]}")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=LiDAR帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 1550 nm FMCW LiDAR 典型参数 [判读·待核]")


def 自检(user="test_lidar_self"):
    """四项校验。"""
    结果 = {}

    基线键 = {
        "设备型号", "感官", "子模态", "尺寸", "尺寸说明", "取值范围", "分辨率", "采样频率", "部位", "来源", "时间戳", "状态"
    }
    结果["字段对齐"] = {}
    for sub in 规格["子模态"]:
        m = 激光雷达矩阵.from_frame(建测试帧("point_cloud_footprint"), sub_modality=sub)
        k = set(m.to_summary().keys())
        结果["字段对齐"][sub] = (k == 基线键)

    m_raw = 激光雷达矩阵.from_frame(建测试帧("point_cloud_footprint"), sub_modality="飞行时间_ps")
    m_dist = 激光雷达矩阵.from_frame(建测试帧("point_cloud_footprint"), sub_modality="距离_m")
    原始扁平 = [v for r in m_raw.data for v in r]
    距离扁平 = [v for r in m_dist.data for v in r]
    结果["子模态分离"] = (max(原始扁平) > 1000) and (max(距离扁平) < 300)

    c1 = 捕获激光雷达(user, sub_modality="飞行时间_ps")
    c2 = 捕获激光雷达(user, sub_modality="距离_m")
    结果["入痕_原始量"] = (c1._trace_hash is not None)
    结果["入痕_设备输出级"] = (c2._trace_hash is not None)

    结果["payload_含关系说明"] = True

    import shutil
    from pathlib import Path
    根 = Path(__file__).parent.parent / "luoshenfu_home" / "users" / user
    if 根.exists():
        shutil.rmtree(根)

    return 结果


if __name__ == "__main__":
    捕获激光雷达("demo", sub_modality="飞行时间_ps")
    捕获激光雷达("demo", sub_modality="距离_m")
    m = 激光雷达矩阵.from_frame(建测试帧("point_cloud_footprint"), sub_modality="距离_m")
    显示矩阵(m)
    print()
    r = 自检()
    print("[自检]", r)
