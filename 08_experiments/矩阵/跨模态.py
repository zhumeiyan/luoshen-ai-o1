"""跨模态差分原语 v0.3 — 帧差分 + 跨模态差分（六维特征版）。

接 between.py（实际在 luoshenfu.py）的 Difference / Pair / CCoord 框架。

v0.3 改动（自 v0.2）：
  - 特征：[mean, max]（2D）→ [mean, max, skew, kurt, p25, p75]（6D）
  - 归一化：保留 L2，泛化到 N 维
  - 中心化/步进/反称：泛化到 N 维

v0.2 改动（自 v0.1 归档版）：
  - 归一化：L∞ → L2（保留方向，不坍缩维度）
  - 全路自净后进入差分（温度去基线、痛觉转调制、稀疏路标覆盖度）
  - 时间基对齐（缺席标记，不填零）

两种差分：
  帧差分（帧差分）
    对象：同一通道相邻两帧的矩阵值差分
    运行：提取 6D 特征 → L2 归一化 → Difference 写入 → 算 g
    含义：同一通道内容随时间变化了多少

  跨模态差分（跨模态差分）
    对象：两个不同通道同一时刻的 6D 特征差分
    运行：L2 归一化 → 中心化 → Pair(差_a, 差_b) 落入反称 → 算 g / h / C坐标
    含义：两个通道在同一瞬时的"差距"及其在 C 坐标系的位置

纪律：
  - 不做语义解释/意义推断
  - 只做"矩阵 → 数值特征 → 差分 → 入痕"
  - 纯数值，LLM 不参与
  - [判读·待核]

设计选择（必须标清，非自然成立）：
  1. 特征提取：6D = [mean, max, skewness, kurtosis, p25, p75]，描述分布形状
  2. 归一化：L2 归一化，N 维方向保留
  3. 中心化：a_bias = 规一_a - center, b_bias = 规一_b - center → a_bias + b_bias = 0
  4. g/h 分离：创建 Pair 后全维等幅步进，让轨迹偏离 bias，促成 g ≠ h
  5. γ（"差异相对量"）：γ = |g| / σ，γ ≠ between.py α
  6. 六维承载：skew/kurt/p25/p75 各维携带不同分布形状信息
"""

from 洛神 import Difference, Pair, CCoord, Trace


def _分位数(排序数据, p):
    """线性插值分位数。"""
    if not 排序数据:
        return 0.0
    k = (len(排序数据) - 1) * p
    f = int(k)
    c = min(f + 1, len(排序数据) - 1)
    d = k - f
    return 排序数据[f] + d * (排序数据[c] - 排序数据[f])


def 矩阵转六维(矩阵):
    """把任意矩阵压成 6D 特征向量 [mean, max, skew, kurt, p25, p75]。

    六维描述分布形状：
      mean     → 分布中心
      max      → 分布尾部极端值
      skew     → 偏度（正=右偏，负=左偏，0=对称）
      kurt     → 峰度（超值峰度，正=尖峰，负=平坦）
      p25/p75  → 分布跨度的上下四分位
    """
    扁平 = sorted([v for row in 矩阵 for v in row])
    if not 扁平:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    n = len(扁平)
    均值 = sum(扁平) / n
    最大 = 扁平[-1]

    方差 = sum((v - 均值) ** 2 for v in 扁平) / n
    标准差 = 方差 ** 0.5

    if 标准差 < 1e-12:
        偏度 = 0.0
        峰度 = 0.0
    else:
        偏度 = sum((v - 均值) ** 3 for v in 扁平) / (n * 标准差 ** 3)
        峰度 = sum((v - 均值) ** 4 for v in 扁平) / (n * 标准差 ** 4) - 3.0

    p25 = _分位数(扁平, 0.25)
    p75 = _分位数(扁平, 0.75)

    return [均值, 最大, 偏度, 峰度, p25, p75]


def 归一化N维(特征):
    """用 L2 范数归一化到 N 维单位球面（保留方向）。

    N 维通用，不依赖维度数。
    不同分布形状在 N 维单位球面上方向不同 → 可区分。
    """
    if not 特征:
        return 特征
    模长 = (sum(v ** 2 for v in 特征)) ** 0.5
    if 模长 < 1e-12:
        return [0.0] * len(特征)
    return [v / 模长 for v in 特征]


def 算覆盖度(矩阵):
    """计算稀疏路覆盖率（非零像素 / 总像素）。"""
    扁平 = [v for row in 矩阵 for v in row]
    if not 扁平:
        return 0.0
    非零 = sum(1 for v in 扁平 if abs(v) > 1e-10)
    return 非零 / len(扁平)


def 去基线(矩阵, 基线):
    """自净：去基线（温度等缓变环境基线）。"""
    return [[v - 基线 for v in row] for row in 矩阵]


def 帧差分(user, 矩阵_t0, 矩阵_t1, 通道="视觉", 基线=None):
    """帧差分：同一通道相邻两帧的差。

    可选自净：基线 非 None 时先去基线。
    L2 归一化（6D 特征）。
    """
    m0 = 矩阵_t0
    m1 = 矩阵_t1
    if 基线 is not None:
        m0 = 去基线(矩阵_t0, 基线)
        m1 = 去基线(矩阵_t1, 基线)

    特征0 = 矩阵转六维(m0)
    特征1 = 矩阵转六维(m1)

    归一0 = 归一化N维(特征0)
    归一1 = 归一化N维(特征1)

    d = Difference(f"帧差_{通道}", bias0=归一0)
    步进 = [归一1[i] - 归一0[i] for i in range(len(特征0))]
    d.step(步进)

    g_vec = d.trajectory[-1]
    prev = d.trajectory[-2] if len(d.trajectory) >= 2 else [0.0] * len(特征0)
    实际差 = [g_vec[i] - prev[i] for i in range(len(特征0))]

    模g = (sum(v ** 2 for v in g_vec)) ** 0.5
    gamma = 模g  # 帧差分无衰减 σ

    t = Trace(user)
    payload = {
        "type":         "差分",
        "差分类型":     "帧差分",
        "通道":         通道,
        "原始_t0":      特征0,
        "原始_t1":      特征1,
        "归一化_t0":    归一0,
        "归一化_t1":    归一1,
        "差分值":       实际差,
        "步进":         步进,
        "C坐标_σ":      None,
        "C坐标_g":      g_vec,
        "C坐标_h":      None,
        "γ_差异相对量":  gamma,
        "状态":         "[判读·待核]"
    }
    rec = t.append("帧差分", payload, trigger="system")
    return {
        "通道": 通道,
        "特征0": 特征0,
        "特征1": 特征1,
        "步进": 步进,
        "g": g_vec,
        "trajectory": d.trajectory,
        "diff": 实际差,
        "gamma": gamma,
        "trace_hash": rec["hash"][:12],
        "trace_seq": rec["seq"]
    }


def 跨模态差分(user, 矩阵_a, 矩阵_b, 通道_a, 通道_b,
             漏出=0.99, 覆盖度_a=None, 覆盖度_b=None,
             角色_a="evidence", 角色_b="evidence"):
    """跨模态差分：两个通道同一时刻的差异 → Pair / Between。

    v0.3（六维特征版）：
    - 6D 特征 → L2 → 6D 中心化/步进/反称
    - 维度数由特征向量长度自动决定

    参数：
      覆盖度_a/b：稀疏路覆盖率（0..1），None 标 1
      角色_a/b：evidence / modulation / substrate / sparse
    """
    特征_a = 矩阵转六维(矩阵_a)
    特征_b = 矩阵转六维(矩阵_b)

    # ── 1. L2 归一化（N 维） ──
    规一_a = 归一化N维(特征_a)
    规一_b = 归一化N维(特征_b)

    # ── 2. 中心化 → L2 反称（N 维） ──
    维数 = len(规一_a)
    中心 = [(规一_a[i] + 规一_b[i]) / 2.0 for i in range(维数)]
    a_偏置 = [规一_a[i] - 中心[i] for i in range(维数)]
    b_偏置 = [规一_b[i] - 中心[i] for i in range(维数)]

    s = [a_偏置[i] + b_偏置[i] for i in range(维数)]
    assert all(abs(s[i]) < 1e-12 for i in range(维数)), f"反称失败: {s}"

    a = Difference(f"跨模_{通道_a}", bias0=a_偏置)
    b = Difference(f"跨模_{通道_b}", bias0=b_偏置)

    # ── 3. 步进 → g ≠ h 分离（N 维等幅步进） ──
    步幅 = 0.01
    a.step([+步幅] * 维数)
    b.step([-步幅] * 维数)

    pair = Pair(a, b)
    g, h = pair.g(), pair.h()

    cc = CCoord(sigma=1.0, leak=漏出)
    sig, _, _ = cc.step(g, h)

    模g = (sum(v ** 2 for v in g)) ** 0.5
    gamma = 模g / sig if sig > 0 else None

    cov_a = 覆盖度_a if 覆盖度_a is not None else 1.0
    cov_b = 覆盖度_b if 覆盖度_b is not None else 1.0

    t = Trace(user)
    payload = {
        "type":         "差分",
        "差分类型":     "跨模态差分",
        "通道A":        通道_a,
        "通道B":        通道_b,
        "原始_特征_a":  特征_a,
        "原始_特征_b":  特征_b,
        "归一化_a":     规一_a,
        "归一化_b":     规一_b,
        "中心":         中心,
        "a_偏置":      a_偏置,
        "b_偏置":      b_偏置,
        "差分值_g":     g,
        "锚差_h":       h,
        "C坐标_σ":      sig,
        "C坐标_g":      g,
        "C坐标_h":      h,
        "γ_差异相对量": gamma,
        "覆盖度_a":     cov_a,
        "覆盖度_b":     cov_b,
        "角色_a":       角色_a,
        "角色_b":       角色_b,
        "特征维度":     维数,
        "状态":         "[判读·待核]"
    }
    rec = t.append("跨模态差分", payload, trigger="system")
    return {
        "通道_a": 通道_a,
        "通道_b": 通道_b,
        "特征_a": 特征_a,
        "特征_b": 特征_b,
        "规一_a": 规一_a,
        "规一_b": 规一_b,
        "中心": 中心,
        "a_偏置": a_偏置,
        "b_偏置": b_偏置,
        "g": g,
        "h": h,
        "sigma": sig,
        "gamma": gamma,
        "覆盖度_a": cov_a,
        "覆盖度_b": cov_b,
        "角色_a": 角色_a,
        "角色_b": 角色_b,
        "特征维度": 维数,
        "trace_hash": rec["hash"][:12],
        "trace_seq": rec["seq"]
    }


def 跑最小演示(user="cross_modal_demo"):
    """最小演示：六维特征跨模态差分。"""
    from 矩阵.视觉 import 建测试帧, 视觉矩阵
    from 矩阵.听觉 import 建测试帧 as 听觉帧, 听矩阵

    m0 = 视觉矩阵.from_frame(建测试帧(花样="gradient")).data
    aud_mat = 听矩阵.from_frame(听觉帧(花样="tone_sweep")).data
    cd = 跨模态差分(user, m0, aud_mat, "视觉", "听觉")
    return cd


if __name__ == "__main__":
    cd = 跑最小演示()
    print("=== 跨模态差分 v0.3 (6D) ===")
    特征名 = ["mean", "max", "skew", "kurt", "p25", "p75"]
    print(f"\n特征维度数: {cd['特征维度']}")
    print(f"视觉 6D 原始: {dict(zip(特征名, [f'{v:.4f}' for v in cd['特征_a']]))}")
    print(f"视觉 6D L2:   {dict(zip(特征名, [f'{v:.4f}' for v in cd['规一_a']]))}")
    print(f"听觉 6D 原始: {dict(zip(特征名, [f'{v:.4f}' for v in cd['特征_b']]))}")
    print(f"听觉 6D L2:   {dict(zip(特征名, [f'{v:.4f}' for v in cd['规一_b']]))}")
    print(f"\ng={cd['g']}")
    print(f"|g|={sum(v**2 for v in cd['g'])**0.5:.6f}")
    print(f"γ={cd['gamma']:.6f}")
