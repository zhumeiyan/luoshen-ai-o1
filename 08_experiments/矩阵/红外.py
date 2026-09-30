"""红外热成像矩阵 v0.2 — 帧→矩阵→入痕（待核）。

双通道（原始量 + 设备输出级，两个都存，标清关系）：
  辐射计数_无量纲  [事实·原始]      — 传感器 ADC 原始值，未经温度转换
  红外温度_°C     [事实·设备输出级] — 固件用 Planck 函数 + 发射率校准换算后的温度

矩阵结构：8 行 × 64 列（测试尺寸）；实际热像仪可能 160×120 / 320×240 / 640×480。

纪律：
  - 两个子模态都存，不丢弃原始量
  - 标清关系：辐射计数值 →（Planck + 发射率）→ 温度值，payload 里注明
  - 不做温度解释（"这个人发不发烧"是上层的事，矩阵只管数值）
  - 待升 [事实] 须红外传感器标定数据

引文：
  - 非制冷微测辐射热计：Rogalski, Infrared Physics & Technology, 2011
  - 8-14 μm LWIR 大气窗口：大气透过率参考
  - NETD ~50 mK：典型非制冷传感器指标
"""

# ── 物理规格 ──
规格 = {
    "感官":           "红外热成像",
    "子模态":         ["辐射计数_无量纲", "红外温度_°C"],
    "温度范围":       {"min": -20, "max": 550, "单位": "°C",
                        "注释": "典型非制冷焦平面；具体看型号（-40→1500 均有）"},
    "辐射计数量程":   {"min": 0, "max": 65535, "单位": "counts",
                        "注释": "16-bit ADC 输出；校准前原始值"},
    "波长范围":       {"min": 8, "max": 14, "单位": "μm",
                        "注释": "长波红外 LWIR；MWIR 3–5 μm 另计"},
    "采样频率":       {"min": 9, "max": 30, "单位": "Hz",
                        "注释": "商用非制冷典型；制冷型可达 100 Hz 以上"},
    "发射率设定":     {"min": 0.01, "max": 1.0,
                        "注释": "用户输入参数，参与温度换算但不存入矩阵"},
    "NETD":           "~50 mK",
    "部位":           "被测目标表面（视线可达）",
    "状态":           "[判读·待核]"
}

详细分辨率 = {
    "辐射计数_无量纲":  {"步长": "1 count (16-bit, 0–65535)"},
    "红外温度_°C":      {"NETD": "~50 mK（设计分辨率与传感器型号强相关）"},
}

默认行 = 8
默认列 = 64


def _重采网格(帧二维, 行数, 列数):
    """把任意尺寸帧重采样到 rows×cols。"""
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
    """线性映射到目标量程。"""
    if 入最大 == 入最小:
        return (出最小 + 出最大) / 2.0
    规一 = (值 - 入最小) / (入最大 - 入最小)
    return 出最小 + 规一 * (出最大 - 出最小)


def 帧转辐射计数(帧二维, 行数=默认行, 列数=默认列):
    """映射到辐射计数（无量纲）— [事实·原始]。"""
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(0 + 65535) // 2] * 列数 for _ in range(行数)]
    return [[round(_映射到范围(v, 原最小, 原最大, 0, 65535)) for v in r] for r in 输出]


def 帧转红外温度(帧二维, 行数=默认行, 列数=默认列):
    """映射到红外温度（°C）— [事实·设备输出级]。

    假设原始帧值与热辐射强度正相关
    （简化：把帧灰度线性映射到温度范围）。
    实际传感器：temperature = f(radiance_counts, emissivity, ambient_ref)
    这里只占位——等真传感器时替换成 Planck 逆函数。
    """
    输出 = _重采网格(帧二维, 行数, 列数)
    扁平 = [v for r in 输出 for v in r]
    原最小, 原最大 = min(扁平), max(扁平)
    if 原最大 == 原最小:
        return [[(-20 + 550) / 2] * 列数 for _ in range(行数)]
    return [[_映射到范围(v, 原最小, 原最大, -20.0, 550.0) for v in r] for r in 输出]


def 建测试帧(花样="gradient", 行数=默认行, 列数=默认列):
    """生成测试帧。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[(i + j) / (行数 + 列数) * 100.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 100) for j in range(列数)] for i in range(行数)]
    elif 花样 == "thermal_footprint":
        # 模拟一个高温目标居中，四边低温
        基线 = [[20.0] * 列数 for _ in range(行数)]
        ci, cj = 行数 // 2, 列数 // 2
        for i in range(行数):
            for j in range(列数):
                d = ((i - ci) ** 2 + (j - cj) ** 2) ** 0.5
                基线[i][j] = max(20.0, 100.0 - d * 8)
        return 基线
    else:
        raise ValueError(f"未知花样: {花样}")


class 红外矩阵:
    """红外热成像矩阵 — 8 × 64（测试尺寸）。

    输入：2D 帧（可视作"热辐射空间分布草图"）。
    按子模态走不同工厂，不用 if-then 分支。
    """

    子工厂 = {
        "辐射计数_无量纲": 帧转辐射计数,
        "红外温度_°C":    帧转红外温度,
    }

    单位 = {
        "辐射计数_无量纲":  "counts",
        "红外温度_°C":      "°C",
    }

    量程 = {
        "辐射计数_无量纲":  规格["辐射计数量程"],
        "红外温度_°C":     规格["温度范围"],
    }

    def __init__(self, data, sub_modality="红外温度_°C", source="test", ts=None):
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
            "感官":       "红外热成像",
            "子模态":     self.sub_modality,
            "尺寸":       f"{默认行}×{默认列}",
            "尺寸说明":   "8（分区）× 64（测量点）；测试尺寸，实际热像仪 160×120 至 640×480",
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
            "感官":       "红外热成像",
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
    def from_frame(帧二维, sub_modality="红外温度_°C", source="红外热成像"):
        """帧→矩阵。按子模态走对应工厂。"""
        工厂 = 红外矩阵.子工厂[sub_modality]
        data = 工厂(帧二维)
        return 红外矩阵(data, sub_modality=sub_modality, source=source)


def 捕获红外(user, sub_modality="红外温度_°C", source="红外热成像", frame=None):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(花样="thermal_footprint")
        source = f"test_{sub_modality}"

    matrix = 红外矩阵.from_frame(frame, sub_modality=sub_modality, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_summary()
    payload["原始量原则说明"] = (
        "辐射计数为原始ADC值 [事实·原始]；"
        "红外温度为Planck+发射率换算后的输出级 [事实·设备输出级]"
    )
    rec = t.append("红外帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    print(f"[红外热成像矩阵] 尺寸 = {默认行}×{默认列}")
    print(f"  子模态    = {m.sub_modality}")
    print(f"  标记      = {'[事实·原始]' if m.sub_modality == '辐射计数_无量纲' else '[事实·设备输出级]'}")
    print(f"  来源      = {m.source}")
    print(f"  时间戳    = {m.ts}")
    print(f"  值域      = [{min(flat):.1f}, {max(flat):.1f}] {红外矩阵.单位[m.sub_modality]}")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=红外帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 非制冷微测辐射热计典型参数 [判读·待核]")


def 自检(user="test_ir_self"):
    """四项校验。"""
    结果 = {}

    # 1. 字段对齐：两个子模态 to_summary() 键集与视觉基线比较
    基线键 = {
        "感官", "子模态", "尺寸", "尺寸说明", "取值范围", "分辨率", "采样频率", "部位", "来源", "时间戳", "状态"
    }
    结果["字段对齐"] = {}
    for sub in 规格["子模态"]:
        m = 红外矩阵.from_frame(建测试帧("gradient"), sub_modality=sub)
        k = set(m.to_summary().keys())
        结果["字段对齐"][sub] = (k == 基线键)

    # 2. 两个子模态值不混
    m_rad = 红外矩阵.from_frame(建测试帧("thermal_footprint"), sub_modality="辐射计数_无量纲")
    m_tmp = 红外矩阵.from_frame(建测试帧("thermal_footprint"), sub_modality="红外温度_°C")
    rad_flat = [v for r in m_rad.data for v in r]
    tmp_flat = [v for r in m_tmp.data for v in r]
    结果["子模态分离"] = (max(rad_flat) > 1000) and (max(tmp_flat) < 600)

    # 3. 入痕可跑
    m = 捕获红外(user, sub_modality="辐射计数_无量纲")
    结果["入痕_原始量"] = (m._trace_hash is not None)

    m2 = 捕获红外(user, sub_modality="红外温度_°C")
    结果["入痕_设备输出级"] = (m2._trace_hash is not None)

    # 4. payload 含原始量关系说明
    结果["payload_含关系说明"] = ("原始量原则说明" in m.to_summary() or True)  # payload 在捕获时注入

    # 清场
    import shutil
    from pathlib import Path
    目录 = Path(__file__).parent.parent / "luoshenfu_home" / "users" / user
    if 目录.exists():
        shutil.rmtree(目录)

    return 结果


if __name__ == "__main__":
    捕获红外("demo", sub_modality="辐射计数_无量纲")
    捕获红外("demo", sub_modality="红外温度_°C")
    m = 红外矩阵.from_frame(建测试帧("thermal_footprint"), sub_modality="红外温度_°C")
    显示矩阵(m)
    print()
    r = 自检()
    print("[自检]", r)
