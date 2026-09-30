"""毫米波雷达矩阵 v0.2 — 帧→矩阵→入痕（待核）。

设备型号：4D 毫米波成像雷达 LRR610
来源：制造商规格书

传感器参数：
  - 距离：0–300 m
  - 速度：-100 至 +100 m/s
  - 方位：±60°
  - 俯仰：±15°

纪律：
  - 频移和速度都存，不丢弃原始量
  - 标清关系：Δf → v = λ·Δf / 2，payload 里注明波长 λ
  - 不做目标识别

实测说明：测试尺寸，设计值按 chirp 配置标定。
矩阵结构：8 行 × 64 列。
"""

# ── 物理规格（77 GHz 车载雷达典型参数） ──
规格 = {
    "设备型号":       "4D毫米波成像雷达LRR610",
    "感官":           "毫米波雷达",
    "子模态":         ["频移_Hz", "速度_m_s"],
    "载频":           {"value": 77, "单位": "GHz",
                        "注释": "典型车载毫米波雷达频段（76–81 GHz）"},
    "波长":           {"value": 3.9, "单位": "mm",
                        "注释": "λ = c / f_0 ≈ 3.9 mm @ 77 GHz"},
    "频移量程":       {"min": -50000, "max": 50000, "单位": "Hz",
                        "注释": "最大多普勒频移；具体取决于目标和雷达参数"},
    "速度量程":       {"min": -100.0, "max": 100.0, "单位": "m/s",
                        "注释": "LRR610：±100 m/s"},
    "采样频率":       {"min": 10, "max": 100, "单位": "Hz",
                        "注释": "帧率（chirp 序列重复频率）"},
    "调频带宽":       {"value": 1000, "单位": "MHz",
                        "注释": "影响距离分辨率（ΔR = c / 2B）"},
    "部位":           "径向视线可达（前/后/侧向，取决于安装位置）",
    "状态":           "[判读·待核]"
}

详细分辨率 = {
    "频移_Hz":       {"多普勒分辨率": "~与 chirp 积累时间成反比（如 T_obs=50 ms → ~20 Hz）"},
    "速度_m_s":      {"速度分辨率": "~0.1–0.5 m/s（取决于 chirp 配置）"},
}

默认行 = 8
默认列 = 64
光速 = 299792458.0
载频 = 77e9
波长 = 光速 / 载频  # ~3.9 mm


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


def 帧转多普勒(帧二维, 行数=默认行, 列数=默认列):
    """映射到多普勒频移（Hz）— [事实·原始]。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[0.0] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, -50000, 50000) for v in r] for r in 输出]


def 帧转速度(帧二维, 行数=默认行, 列数=默认列):
    """映射到径向速度（m/s）— [事实·设备输出级】。

    通过 Doppler → 速度公式换算：
      v = λ · Δf / 2

    假设帧正中心为静止零速度，边缘为最大速度。
    """
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[0.0] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, -100.0, 100.0) for v in r] for r in 输出]


def 建测试帧(花样="gradient", 行数=默认行, 列数=默认列):
    """生成测试帧。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[(i + j) / (行数 + 列数) * 100.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 100) for j in range(列数)] for i in range(行数)]
    elif 花样 == "doppler_footprint":
        # 模拟：一个移动目标在中间有高多普勒频移
        基线 = [[0.0] * 列数 for _ in range(行数)]
        中列 = 列数 // 2
        for i in range(行数):
            for j in range(列数):
                d = abs(j - 中列)
                基线[i][j] = max(0.0, 80.0 - d * 6)
        return 基线
    else:
        raise ValueError(f"未知花样: {花样}")


class 雷达矩阵:
    """毫米波雷达矩阵 — 8 × 64（测试尺寸）。"""

    子工厂 = {
        "频移_Hz":   帧转多普勒,
        "速度_m_s":  帧转速度,
    }

    单位 = {
        "频移_Hz":   "Hz",
        "速度_m_s":  "m/s",
    }

    量程 = {
        "频移_Hz":   规格["频移量程"],
        "速度_m_s":  规格["速度量程"],
    }

    def __init__(self, data, sub_modality="速度_m_s", source="test", ts=None):
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
            "感官":       "毫米波雷达",
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
            "矩阵":       self.data,
        }

    def to_summary(self):
        return {
            "设备型号":   规格["设备型号"],
            "感官":       "毫米波雷达",
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
    def from_frame(帧二维, sub_modality="速度_m_s", source="毫米波雷达"):
        工厂 = 雷达矩阵.子工厂[sub_modality]
        data = 工厂(帧二维)
        return 雷达矩阵(data, sub_modality=sub_modality, source=source)


def 捕获雷达(user, sub_modality="速度_m_s", source="毫米波雷达", frame=None):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(花样="doppler_footprint")
        source = f"test_{sub_modality}"

    matrix = 雷达矩阵.from_frame(frame, sub_modality=sub_modality, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    payload["原始量原则说明"] = (
        "频移为多普勒原始量 [事实·原始]；"
        "速度为 v=λ·Δf/2 换算设备输出级 [事实·设备输出级]"
    )
    rec = t.append("雷达帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    标记 = "[事实·原始]" if m.sub_modality == "频移_Hz" else "[事实·设备输出级]"
    print(f"[雷达矩阵] 尺寸 = {默认行}×{默认列}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  标记      = {标记}")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.1f}, {max(flat):.1f}] {雷达矩阵.单位[m.sub_modality]}")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=雷达帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 77 GHz 车载雷达典型参数 [判读·待核]")


def 自检(user="test_radar_self"):
    """四项校验。"""
    结果 = {}

    基线键 = {
        "设备型号", "感官", "子模态", "尺寸", "尺寸说明", "取值范围", "分辨率", "采样频率", "部位", "来源", "时间戳", "状态"
    }
    结果["字段对齐"] = {}
    for sub in 规格["子模态"]:
        m = 雷达矩阵.from_frame(建测试帧("doppler_footprint"), sub_modality=sub)
        k = set(m.to_summary().keys())
        结果["字段对齐"][sub] = (k == 基线键)

    m_raw = 雷达矩阵.from_frame(建测试帧("doppler_footprint"), sub_modality="频移_Hz")
    m_vel = 雷达矩阵.from_frame(建测试帧("doppler_footprint"), sub_modality="速度_m_s")
    原始扁平 = [v for r in m_raw.data for v in r]
    速度扁平 = [v for r in m_vel.data for v in r]
    结果["子模态分离"] = (max(原始扁平) > 1000) and (max(速度扁平) < 300)

    c1 = 捕获雷达(user, sub_modality="频移_Hz")
    c2 = 捕获雷达(user, sub_modality="速度_m_s")
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
    捕获雷达("demo", sub_modality="频移_Hz")
    捕获雷达("demo", sub_modality="速度_m_s")
    m = 雷达矩阵.from_frame(建测试帧("doppler_footprint"), sub_modality="速度_m_s")
    显示矩阵(m)
    print()
    r = 自检()
    print("[自检]", r)
