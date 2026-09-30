"""对话测试 v0.2 — 词典短路 + 自净 + 复核。

链路（v0.2 新增词典短路）：
  文本输入
    → 拆字
    → 词典层：逐字查字典
        · 命中 → 定案（认出来了）
        · 未命中 → 标"未见"
    → 连续层（γ融合）：只做复核
        · 对得上 → 确认
        · 对不上 → 标"破绽"
    → 输出：词典结果 + 复核结果
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from 矩阵.跨模态 import (
    跨模态差分, 归一化N维, 矩阵转六维,
    compute_coverage, remove_baseline
)


# ────────────────────────────────────────
# 第一件：词典
# ────────────────────────────────────────

CHAR_DICTIONARY = {
    # 常用字 + 测试用字
    "你": {"pinyin": "ni3",  "category": "代词", "frequency": "high"},
    "好": {"pinyin": "hao3",  "category": "形容词", "frequency": "high"},
    "洛": {"pinyin": "luo4",  "category": "名词", "frequency": "mid"},
    "神": {"pinyin": "shen2", "category": "名词", "frequency": "high"},
    "山": {"pinyin": "shan1", "category": "名词", "frequency": "high"},
    "水": {"pinyin": "shui3", "category": "名词", "frequency": "high"},
    "天": {"pinyin": "tian1", "category": "名词", "frequency": "high"},
    "地": {"pinyin": "di4",   "category": "名词", "frequency": "high"},
    "人": {"pinyin": "ren2",  "category": "名词", "frequency": "high"},
    "我": {"pinyin": "wo3",   "category": "代词", "frequency": "high"},
    "他": {"pinyin": "ta1",   "category": "代词", "frequency": "high"},
    "她": {"pinyin": "ta1",   "category": "代词", "frequency": "high"},
    "是": {"pinyin": "shi4",  "category": "动词", "frequency": "high"},
    "不": {"pinyin": "bu4",   "category": "副词", "frequency": "high"},
    "了": {"pinyin": "le",    "category": "助词", "frequency": "high"},
    "的": {"pinyin": "de",    "category": "助词", "frequency": "high"},
    "在": {"pinyin": "zai4",  "category": "介词", "frequency": "high"},
    "有": {"pinyin": "you3",  "category": "动词", "frequency": "high"},
    "这": {"pinyin": "zhe4",  "category": "代词", "frequency": "high"},
    "那": {"pinyin": "na4",   "category": "代词", "frequency": "high"},
    "大": {"pinyin": "da4",   "category": "形容词", "frequency": "high"},
    "小": {"pinyin": "xiao3", "category": "形容词", "frequency": "high"},
    "中": {"pinyin": "zhong1","category": "方位", "frequency": "high"},
    "上": {"pinyin": "shang4","category": "方位", "frequency": "high"},
    "下": {"pinyin": "xia4",  "category": "方位", "frequency": "high"},
    "左": {"pinyin": "zuo3",  "category": "方位", "frequency": "high"},
    "右": {"pinyin": "you4",  "category": "方位", "frequency": "high"},
    "日": {"pinyin": "ri4",   "category": "名词", "frequency": "high"},
    "月": {"pinyin": "yue4",  "category": "名词", "frequency": "high"},
    "风": {"pinyin": "feng1", "category": "名词", "frequency": "high"},
    "雨": {"pinyin": "yu3",   "category": "名词", "frequency": "high"},
    "心": {"pinyin": "xin1",  "category": "名词", "frequency": "high"},
    "意": {"pinyin": "yi4",   "category": "名词", "frequency": "high"},
}


def dictionary_lookup(char):
    """词典层：单个汉字查词典。"""
    if char in CHAR_DICTIONARY:
        return {
            "hit": True,
            "char": char,
            "pinyin": CHAR_DICTIONARY[char]["pinyin"],
            "category": CHAR_DICTIONARY[char]["category"],
            "frequency": CHAR_DICTIONARY[char]["frequency"],
            "source": "词典"
        }
    else:
        return {
            "hit": False,
            "char": char,
            "pinyin": None,
            "category": None,
            "frequency": None,
            "source": "未见"
        }


def dictionary_recognize(text):
    """词典层：整句逐字查词典。"""
    results = []
    for char in text:
        results.append(dictionary_lookup(char))
    return results


# ────────────────────────────────────────
# 第二件：输入转换（明报）
# ────────────────────────────────────────

def text_to_seed(text):
    """输入转换方式：文本 → 种子码点。

    规则：
      1. 取每个字符的 Unicode code point（ord）
      2. 对 256 取模后 +1，映射到 [1, 256]
      3. 输出为整数列表

    示例："你好洛神" → [97, 126, 28, 95]
      '你'=U+4F60=20320, 20320%256=96, +1=97
      '好'=U+597D=22909, 22909%256=125, +1=126
      '洛'=U+6D1C=27932, 27932%256=27,  +1=28
      '神'=U+795E=31070, 31070%256=94,  +1=95

    性质：
      - 确定性：同一字每次出相同码点
      - 非语义：码点不含字义信息，仅作矩阵种子
      - 分布：模 256 后散落，非保序
    """
    return [ord(c) % 256 + 1 for c in text]


def seed_to_matrix(name, seed, rows=8, cols=64):
    """种子 → 单路矩阵。

    各通道规则（不同通道用不同变换，模拟不同物理传感器对不同刺激的响应）：

      视觉：每字码点 × 1000 → 亮度（cd/m²）
            公式：matrix[i] = seed[i] × 1000
            无字处：随机填充 [10, 500]（模拟背景亮度）
            覆盖率：≈100%

      听觉：码点 × 5 + 80 → 基频（Hz）
            公式：matrix[i] = 80 + seed[i] × 5
            无字处：0（静默）
            覆盖率：≈ 字数/512

      触觉：码点 × 4 → 压力（kPa）
            公式：matrix[i] = seed[i] × 4
            无字处：0（无接触）

      温度觉：环境基线 293K + 随机偏移 [-5, +15] → 温度（°C）
            公式：matrix[i] = 293 + U(-5, 15)
            无字处：同样有值（温度场连续分布，无"缺席"）

      痛觉：码点 ÷ 25.6 → 痛觉强度（0~10）
            公式：matrix[i] = seed[i] / 25.6
            无字处：0（无痛刺激）

      发声：码点 × 3 + 200 → 基频（Hz）
            公式：matrix[i] = 200 + seed[i] × 3
            无字处：0（无声）

      红外：码点 × 2 + 200 → 红外强度
            公式：matrix[i] = 200 + seed[i] × 2
            无字处：0（无红外源）

      雷达：码点 > 100 时出速度值（m/s）
            公式：若 seed[i] > 100, matrix[i] = (seed[i]-100)/156 × 200 - 100
                  否则 0
            无字处：0（无目标）

      LiDAR：码点 mod 3 == 0 时出距离值（m）
            公式：若 seed[i] % 3 == 0, matrix[i] = 0.3 + seed[i]/255 × 100
                  否则 0
            无字处：0（无回波）

    来源：这些转换规则是设计设定（非自然物理），
          目的是让不同通道对同一输入产生不同分布形状，
          使后续跨模态差分能捕捉分布差异。
    """
    n = rows * cols
    arr = np.zeros(n)

    if name == "视觉":
        for i, v in enumerate(seed):
            arr[i % n] = v * 1000.0
        for i in range(n):
            if arr[i] == 0:
                arr[i] = np.random.uniform(10, 500)

    elif name == "听觉":
        for i, v in enumerate(seed):
            arr[i % n] = 80.0 + v * 5.0

    elif name == "触觉":
        for i, v in enumerate(seed):
            arr[i % n] = v * 4.0

    elif name == "温度觉":
        for i in range(n):
            arr[i] = 293.0 + np.random.uniform(-5, 15)

    elif name == "痛觉":
        for i, v in enumerate(seed):
            arr[i % n] = v / 25.6

    elif name == "发声":
        for i, v in enumerate(seed):
            arr[i % n] = 200.0 + v * 3.0

    elif name == "红外":
        for i, v in enumerate(seed):
            arr[i % n] = 200.0 + v * 2.0

    elif name == "雷达":
        for i, v in enumerate(seed):
            if v > 100:
                arr[i % n] = (v - 100) / 156.0 * 200.0 - 100.0

    elif name == "LiDAR":
        for i, v in enumerate(seed):
            if v % 3 == 0:
                arr[i % n] = 0.3 + (v / 255.0) * 100.0

    return arr.reshape(rows, cols).tolist()


# ────────────────────────────────────────
# 第三件：自净
# ────────────────────────────────────────

CHANNEL_CONFIG = {
    "视觉":    {"role": "evidence", "baseline": None},
    "听觉":    {"role": "evidence", "baseline": None},
    "触觉":    {"role": "sparse",   "baseline": None},
    "温度觉":  {"role": "evidence", "baseline": 293.0},   # 自净：去环境基线
    "痛觉":    {"role": "modulation", "baseline": None},
    "发声":    {"role": "evidence", "baseline": None},
    "红外":    {"role": "evidence", "baseline": None},
    "雷达":    {"role": "sparse",   "baseline": None},
    "LiDAR":   {"role": "sparse",   "baseline": None},
}


def apply_self_cleaning(matrix, channel_name):
    """自净：返回净后矩阵 + 覆盖率 + 角色。"""
    config = CHANNEL_CONFIG[channel_name]
    role = config["role"]
    baseline = config["baseline"]

    m = matrix
    if baseline is not None:
        m = remove_baseline(matrix, baseline)

    coverage = compute_coverage(m)
    return m, coverage, role


# ──────────────────────────────
# 第四件：整条链路
# ──────────────────────────────

def run_pipeline_v02(text):
    print("=" * 80)
    print("【对话测试 v0.2 — 词典短路 + 自净 + 复核】")
    print("=" * 80)
    print(f"\n『输入』「{text}」")

    # ── Step 1: 词典层 ──
    print(f"\n── 第一层：词典 ──")
    dict_results = dictionary_recognize(text)
    recognized_chars = []
    for r in dict_results:
        status = "命中" if r["hit"] else "未见"
        print(f"  「{r['char']}」 → {status}"
              f"  {r.get('pinyin', '')}  {r.get('category', '')}")
        if r["hit"]:
            recognized_chars.append(r["char"])

    # ── Step 2: 输入转换（明报） ──
    print(f"\n── 第二层：输入转换 ──")
    seed = text_to_seed(text)
    print(f"  规则：ord(c) % 256 + 1 → 码点 [1, 256]")
    print(f"  输出：{seed}")
    print(f"  推导示例：")
    for c in text:
        code = ord(c)
        mapped = code % 256 + 1
        print(f"    '{c}' = U+{code:04X} = {code}, {code}%256={code%256}, +1={mapped}")

    # ── Step 3: 矩阵生成 ──
    print(f"\n── 第三层：种子 → 九路矩阵 ──")
    channels = ["视觉", "听觉", "触觉", "温度觉", "痛觉", "发声", "红外", "雷达", "LiDAR"]
    matrices = {}
    for name in channels:
        matrices[name] = seed_to_matrix(name, seed)

    for name in channels:
        m = matrices[name]
        flat = [v for row in m for v in row]
        print(f"  {name:5s} 值域=[{min(flat):8.1f}, {max(flat):9.1f}]  "
              f"均值={np.mean(flat):8.1f}")

    # ── Step 4: 自净 ──
    print(f"\n── 第四层：自净 ──")
    cleaned = {}
    coverages = {}
    roles = {}
    for name in channels:
        m_clean, cov, role = apply_self_cleaning(matrices[name], name)
        cleaned[name] = m_clean
        coverages[name] = cov
        roles[name] = role
        flat = [v for row in m_clean for v in row]
        print(f"  {name:5s} 自净后值域=[{min(flat):8.2f}, {max(flat):9.2f}]  "
              f"覆盖率={cov:5.1%}  角色={role}")

    # ── Step 5: 六维特征 ──
    print(f"\n── 第五层：六维特征提取 ──")
    feats = {}
    feat_names = ["mean", "max", "skew", "kurt", "p25", "p75"]
    for name in channels:
        f = 矩阵转六维(cleaned[name])
        feats[name] = f
        s = ", ".join(f"{feat_names[i]}={f[i]:.2f}" for i in range(6))
        print(f"  {name:5s} [{s}]")

    # ── Step 6: 36 对跨模态差分 ──
    print(f"\n── 第六层：36 对跨模态差分 ──")
    diff_results = {}
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            na, nb = channels[i], channels[j]
            cd = 跨模态差分(
                "dialog_v02", cleaned[na], cleaned[nb], na, nb,
                coverage_a=coverages[na], coverage_b=coverages[nb],
                role_a=roles[na], role_b=roles[nb]
            )
            diff_results[f"{na}-{nb}"] = cd

    for k, v in sorted(diff_results.items(), key=lambda x: abs(x[1]["gamma"])):
        ga = v["gamma"]
        ga_str = f"{ga:.6f}" if ga is not None else "None"
        print(f"  {k:10s} γ={ga_str:>10s}  覆盖=({v['coverage_a']:4.0%},{v['coverage_b']:4.0%})  "
              f"角色=({v['role_a']:6s},{v['role_b']:6s})")

    # ── Step 7: 融合 → 复核 ──
    print(f"\n── 第七层：融合 / 复核 ──")
    threshold = 0.05
    over = {k: v for k, v in diff_results.items() if v["gamma"] is not None and abs(v["gamma"]) > threshold}
    under = {k: v for k, v in diff_results.items() if v["gamma"] is not None and abs(v["gamma"]) <= threshold}

    print(f"  硬门槛 θ={threshold}")
    print(f"  超过门槛 {len(over)} 对：")
    for k, v in sorted(over.items(), key=lambda x: abs(x[1]["gamma"])):
        print(f"    {k:10s} γ={v['gamma']:+.4f}")
    print(f"  低于门槛 {len(under)} 对：")
    for k, v in sorted(under.items(), key=lambda x: abs(x[1]["gamma"])):
        print(f"    {k:10s} γ={v['gamma']:+.4f}")

    # ── Step 8: 判定 ──
    print(f"\n── 第八层：判定 ──")
    print(f"  词典层：命中 {len(recognized_chars)}/{len(text)} 字 → {recognized_chars}")
    if len(recognized_chars) == len(text):
        # 所有字都在词典中，连续层复核
        print(f"  全部命中 → 连续层复核")

        # 复核：跟文本最相关的几对是否超过门槛
        text_related = {k: v for k, v in over.items()
                        if "视觉" in k or "听觉" in k or "发声" in k}
        temp_radar = {k: v for k, v in over.items()
                      if "温度" in k or "雷达" in k}

        print(f"  文本相关通道分离({len(text_related)} 对)：")
        for k, v in sorted(text_related.items(), key=lambda x: abs(x[1]["gamma"])):
            print(f"    {k:10s} γ={v['gamma']:+.4f}")

        print(f"  环境/设备通道({len(temp_radar)} 对)：")
        for k, v in sorted(temp_radar.items(), key=lambda x: abs(x[1]["gamma"]))[:8]:
            print(f"    {k:10s} γ={v['gamma']:+.4f}")

        if len(text_related) > 0:
            print(f"  → 词典命中 + 文本通道有分离 → 复核通过")
        else:
            print(f"  → 词典命中但文本通道无分离 → 见[破绽]")
    else:
        missing = [c for c in text if c not in recognized_chars]
        print(f"  未全命中 → 缺字：{missing} → 标[未见]")

    print(f"\n状态：[判读·对话测试]")
    return dict_results, matrices, cleaned, diff_results


if __name__ == "__main__":
    dict_results, matrices, cleaned, diff_results = run_pipeline_v02("你好洛神")
    print("\n" + "=" * 80)
    print("\n")
    dict_results2, matrices2, cleaned2, diff_results2 = run_pipeline_v02("地大物博")
