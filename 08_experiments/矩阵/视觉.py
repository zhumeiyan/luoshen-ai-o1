"""视觉矩阵 v0.2 — 帧→矩阵→入痕（待核）。

设备型号：深视智能 sCMOS 科学相机
来源：传感器规格书（读出噪声/满阱/动态范围/像素尺寸/帧率）

传感器参数：
  - 读出噪声：0.45 e⁻
  - 满阱容量：280,000 e⁻
  - 动态范围：115 dB
  - 像素尺寸：2.74 μm
  - 帧率：96 fps @8bit

纪律：
  - 不做图像处理（无边缘检测、无特征提取、无识别）
  - 只做"帧→cd/m²映射→入痕"

实测说明：
  - 48×64 = 测试尺寸（设计值 ≥1080p/4K）
  - 当前数据源为合成测试帧（test_gradient），不是摄像头真帧
  - 时间戳毫秒精度，避免同秒碰撞
  - 字段对齐听觉：13 项字段完全对应
"""

# ── 物理规格（表A·v0.4·视觉行） ──
# 锥密度引文：Curcio et al., J Comp Neurol 1990; 292:497-523. "Human photoreceptor density ~150,000 cones/mm² in fovea"
规格 = {
    "设备型号":   "深视智能sCMOS科学相机",
    "感官":       "视觉",
    "子模态":     ["亮度_cd_m2", "波长_nm"],
    "取值范围":   {"min": 1e-6, "max": 1e5, "单位": "cd/m²"},
    "波长范围":   {"min": 380,  "max": 780,  "单位": "nm"},
    "传感器噪声": {"读出噪声_e": 0.45, "满阱_e": 280000, "动态范围_dB": 115},
    "像素尺寸_um": 2.74,
    "锥密度_mm2": {"中央凹_视锥": "~150,000",
                   "引文": "Curcio et al., J Comp Neurol 1990;292:497-523 (仅视锥, 不含视杆)"},
    "分辨率":     {"对比度": 0.015, "空间": "1角分"},
    "采样频率":   {"min": 30, "max": 96, "单位": "Hz"},
    "部位":       "全视野（中央凹最优）",
    "状态":       "[判读·待核]"
}

# ── 矩阵尺寸 ──
默认行 = 48
默认列 = 64


def 帧转亮度(帧二维, 行数=默认行, 列数=默认列):
    """把任意二维亮度数组映射到 cd/m² 量程。"""
    import math

    if not 帧二维 or not 帧二维[0]:
        raise ValueError("帧二维 为空或第一行为空")

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
    目标最小 = 规格["取值范围"]["min"]
    目标最大 = 规格["取值范围"]["max"]

    if 原最大 == 原最小:
        中 = math.sqrt(目标最小 * 目标最大)
        return [[中] * 列数 for _ in range(行数)], {
            "映射方式": "常量填充（全等输入）"
        }

    映射后 = []
    for r in 输出:
        行 = []
        for v in r:
            规一 = (v - 原最小) / (原最大 - 原最小)
            对数值 = 目标最小 * math.pow(目标最大 / 目标最小, 规一)
            行.append(对数值)
        映射后.append(行)

    return 映射后, {"映射方式": "线性拉伸→对数域"}


def 建测试帧(行数=48, 列数=64, 花样="gradient"):
    """生成测试帧。花样: gradient / checker / random / dark / bright。"""
    import random
    random.seed(42)
    if 花样 == "gradient":
        return [[i / 行数 * 200 for j in range(列数)] for i in range(行数)]
    elif 花样 == "checker":
        return [[((i // 8 + j // 8) % 2) * 255.0 for j in range(列数)] for i in range(行数)]
    elif 花样 == "random":
        return [[random.uniform(0, 255) for j in range(列数)] for i in range(行数)]
    elif 花样 == "dark":
        return [[1e-6 + i * 0.001] * 列数 for i in range(行数)]
    elif 花样 == "bright":
        return [[9e4 - i * 10] * 列数 for i in range(行数)]
    else:
        raise ValueError(f"未知花样: {花样}")


class 视觉矩阵:
    """视觉矩阵 — 48×64 亮度网格（cd/m²）。"""

    def __init__(self, data, source="test", ts=None):
        if len(data) != 默认行 or any(len(r) != 默认列 for r in data):
            raise ValueError(f"矩阵尺寸须为 {默认行}×{默认列}")
        self.data = data
        self.source = source
        # 毫秒精度，避免同秒碰撞
        self.ts = ts or __import__('datetime').datetime.now().isoformat(timespec="milliseconds")

    def to_dict(self):
        return {
            "设备型号": 规格["设备型号"],
            "感官":     "视觉",
            "子模态":   规格["子模态"],
            "尺寸":     f"{默认行}×{默认列}",
            "尺寸说明": "测试尺寸，设计值≥1080p/4K",
            "取值范围": 规格["取值范围"],
            "分辨率":   规格["分辨率"],
            "采样频率": 规格["采样频率"],
            "部位":     规格["部位"],
            "来源":     self.source,
            "时间戳":   self.ts,
            "状态":     "[判读·待核]",
            "矩阵":     self.data,
        }

    def to_summary(self):
        """紧凑版入痕用（不存原始矩阵）。"""
        return {
            "设备型号": 规格["设备型号"],
            "感官":     "视觉",
            "子模态":   规格["子模态"],
            "尺寸":     f"{默认行}×{默认列}",
            "尺寸说明": "测试尺寸，设计值≥1080p/4K",
            "取值范围": 规格["取值范围"],
            "分辨率":   规格["分辨率"],
            "采样频率": 规格["采样频率"],
            "部位":     规格["部位"],
            "来源":     self.source,
            "时间戳":   self.ts,
            "状态":     "[判读·待核]"
        }

    @staticmethod
    def from_frame(帧二维, source="摄像头", 行数=默认行, 列数=默认列):
        data, _ = 帧转亮度(帧二维, 行数, 列数)
        return 视觉矩阵(data, source=source)


def 捕获视觉(user, source="摄像头", frame=None, store_full_matrix=False):
    """帧→矩阵→入痕主入口。"""
    if frame is None:
        frame = 建测试帧(pattern="gradient")
        source = "test_gradient"

    matrix = 视觉矩阵.from_frame(frame, source=source)

    from 洛神 import Trace
    t = Trace(user)
    payload = matrix.to_dict() if store_full_matrix else matrix.to_summary()
    rec = t.append("视觉帧", payload, trigger="system")
    matrix._trace_hash = rec["hash"][:12]
    matrix._trace_seq = rec["seq"]
    return matrix


def 显示矩阵(m):
    flat = [v for r in m.data for v in r]
    print(f"[视觉矩阵]   尺寸 = {默认行}×{默认列}")
    print(f"  来源      = {m.source}")
    print(f"   时间戳  = {m.ts}")
    print(f"  值域      = [{min(flat):.3e}, {max(flat):.3e}] cd/m²  (设计范围 {规格['取值范围']['min']:.0e}–{规格['取值范围']['max']:.0e})")
    if hasattr(m, '_trace_hash'):
        print(f"  入痕      = type=视觉帧, hash={m._trace_hash}, seq={m._trace_seq}")
    print(f"  规格来源  = 表A·v0.4·视觉行 [待核]")
