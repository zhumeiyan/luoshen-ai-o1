"""识人脸.py — 视频 + 话 → 身子 → 看反应是否稳。

流程:
  1. 逐帧取视觉+听觉网格
  2. 每帧算 10 维签名 (识脸签名.全签名, 连续无 if)
  3. 作者均值签名 = 所有帧的几何平均 (连续)
  4. 每帧: 相似度 sim = 1 − ||sig − author_mean|| → 连续 [0,1]
  5. sim 通过 预算增益 注入身子 社交热
  6. 推进后记录 状态
  7. 同一段视频重复 3 次, 每次后"消失" 60 秒墙钟
  8. 跑完后对比三次的 状态 序列, 看是否趋同

规矩:
  - 不加 if (所有分支用 sigmoid/连续加权替代)
  - 不判对错不升主链
  - 真人输入原样进
"""

import sys, os, json, time, math, struct, hashlib, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from 身体稳态 import 身体
from 脸签名 import (
    全签名,
    签名相似度,
)

目录 = Path(__file__).parent
帧目录 = 目录 / "视频输入/帧"
音频路径 = 目录 / "视频输入/音频/语音.wav"
快照路径 = 目录 / "身子状态.json"
报告路径 = 目录 / "识脸报告.tsv"

帧秒 = 10
每帧样本数 = 1600


# ───────────── 手工解码 PPM + WAV ─────────────

def 解像素图(路径):
    with open(路径, "rb") as 文:
        原始 = 文.read()
    头行 = 原始[:200].split(b"\n")
    行号 = 1
    while 头行[行号].strip().startswith(b"#"):
        行号 += 1
    宽, 高 = map(int, 头行[行号].strip().split())
    行号 += 1
    while 头行[行号].strip().startswith(b"#"):
        行号 += 1
    最大值 = int(头行[行号].strip())
    头止 = 原始.index(b"\n", 原始.index(b"\n", 原始.index(头行[0]) + 1) + 1) + 1
    while 原始[头止 : 头止 + 1] == b"\n":
        头止 += 1
    像素数据 = 原始[头止:]
    幻数 = 头行[0].strip()
    行列表 = []
    if 幻数 == b"P3":
        值 = list(map(int, 像素数据.split()))
        for y in range(高):
            行 = [(值[(y * 宽 + x) * 3 + c] for c in range(3)) for x in range(宽)]
            行列表.append(行)
    else:
        for y in range(高):
            行 = [(像素数据[(y * 宽 + x) * 3 + c] for c in range(3)) for x in range(宽)]
            行列表.append(行)
    return 行列表, 高, 宽


def 解波形(路径):
    with open(路径, "rb") as 文:
        数据 = 文.read()
    位 = 0
    while 位 < len(数据) - 8:
        if 数据[位 : 位 + 4] == b"data":
            break
        位 += 1
    位 += 4
    字节数 = struct.unpack_from("<I", 数据, 位)[0]
    位 += 4
    n = 字节数 // 2
    return [struct.unpack_from("<h", 数据, 位 + i * 2)[0] for i in range(n)]


def 网格亮度(图像行, 高, 宽, 行块数=48, 列块数=64):
    网格 = []
    for i in range(行块数):
        行 = []
        for j in range(列块数):
            行起, 行止 = i * 高 // 行块数, max((i + 1) * 高 // 行块数, 1)
            列起, 列止 = j * 宽 // 列块数, max((j + 1) * 宽 // 列块数, 1)
            总 = 0.0
            光 = 0
            for y in range(行起, min(行止, 高)):
                for x in range(列起, min(列止, 宽)):
                    r, g, b = 图像行[y][x]
                    总 += 0.299 * r + 0.587 * g + 0.114 * b
                    光 += 1
            行.append(总 / max(1, 光))
        网格.append(行)
    return 网格


def 网格音频分贝(样本, 行块数=48, 列块数=64):
    """把样本分 64 块 (列), 每块映射到 48 频率行 → 分贝网格。"""
    N = len(样本)
    块大 = max(1, N // 列块数)
    网格 = []
    for _ in range(行块数):
        网格.append([0.0] * 列块数)
    for j in range(列块数):
        块 = 样本[j * 块大 : (j + 1) * 块大]
        if not 块:
            continue
        均方 = math.sqrt(sum(x * x for x in 块) / len(块))
        分贝 = 20.0 * math.log10(均方 + 1.0)
        分贝 = max(0.0, min(60.0, 分贝))
        for i in range(行块数):
            行因子 = 1.0 - abs(i - 12) / 48.0
            行因子 = max(0.05, min(1.0, 行因子))
            网格[i][j] = 分贝 * 行因子
    return 网格


# ───────────── 签名提取 (连续) ─────────────

def 提取签名集():
    """对所有帧提取 10 维签名。"""
    帧序 = sorted(p for p in os.listdir(帧目录) if p.endswith(".ppm"))
    样本 = 解波形(音频路径)
    签名集 = []
    for 序号, 文件名 in enumerate(帧序):
        像素, 高, 宽 = 解像素图(帧目录 / 文件名)
        视觉 = 网格亮度(像素, 高, 宽)
        声起 = 序号 * 每帧样本数
        声止 = 声起 + 每帧样本数
        听觉 = 网格音频分贝(样本[声起:声止])
        签 = 全签名(视觉, 听觉)
        签名集.append(签)
    return 签名集


def 作者均值签名(签名集):
    """几何平均签名 (连续, 无 if) — 对数域。"""
    N = max(1, len(签名集))
    维 = len(签名集[0])
    均 = []
    for k in range(维):
        总 = 0.0
        for 签 in 签名集:
            总 += 签[k] / N
        均.append(总)
    return 均


# ───────────── 注入身子 + 录状态 (连续) ─────────────

def 注入并推进(身子, 签名, 均值签名, 滴答计数):
    """相似度 → 预算增益 (无 if) → 推进 → 返回状态字典。"""
    相似度 = 签名相似度(签名, 均值签名)
    增益 = 相似度 * 0.3
    身子.社交热 = min(1.0, 身子.社交热 + 增益)
    身子.推进()
    return {
        "滴答": 滴答计数,
        "相似度": round(相似度, 4),
        "增益": round(增益, 4),
        "社交热": round(身子.社交热, 4),
        "社交饱": round(getattr(身子, "状态", {}).get("social", 0.0), 4),
        "温度": round(getattr(身子, "状态", {}).get("thermal", 0.0), 3),
        "葡萄糖": round(getattr(身子, "状态", {}).get("glucose", 0.0), 4),
        "声音": round(getattr(身子, "声张", lambda: 0.0)(), 4),
        "签名均值": round(sum(签名) / len(签名), 4),
    }


# ───────────── 主流程 ─────────────

def 跑():
    签名集 = 提取签名集()
    均值签名 = 作者均值签名(签名集)

    # 读身子快照
    import importlib
    import living_daemon as 守
    # 注: 守.快照加载 / 守.身子 / 守.元 等接口待交叉验证
    # 临时用占位身子
    身子 = 身体(节点数=24, 种子=1)
    元 = {"滴答计数": 0}

    # 推进一次取基线
    身子.推进()
    基线 = {
        "社交饱": getattr(身子, "状态", {}).get("social", 0.0),
        "温度": getattr(身子, "状态", {}).get("thermal", 0.0),
        "葡萄糖": getattr(身子, "状态", {}).get("glucose", 0.0),
        "声音": getattr(身子, "声张", lambda: 0.0)(),
    }

    # 3 次脸出现
    全轮状态 = []
    计 = 元.get("滴答计数", 0)
    for 轮 in range(3):
        轮状态 = []
        for 签 in 签名集:
            计 += 1
            状 = 注入并推进(身子, 签, 均值签名, 计)
            轮状态.append(状)
        for _ in range(60):
            计 += 1
            身子.推进()
        全轮状态.append(轮状态)

    # 第四轮: 异脸 (同一张脸 + 连续扰动)
    异脸签名 = [值 * (0.5 + 0.3 * math.sin(i * 0.9)) for i, 值 in enumerate(均值签名)]
    异状态 = []
    for _ in range(len(签名集)):
        计 += 1
        状 = 注入并推进(身子, 异脸签名, 均值签名, 计)
        异状态.append(状)
    for _ in range(60):
        计 += 1
        身子.推进()

    # ───────────── 计算稳度 (连续) ─────────────
    键列 = ["相似度", "社交热", "社交饱"]

    def 序列相似(甲, 乙):
        点 = 0.0
        模甲, 模乙 = 0.0, 0.0
        for i in range(min(len(甲), len(乙))):
            for k in 键列:
                va, vb = 甲[i][k], 乙[i][k]
                点 += va * vb
                模甲 += va * va
                模乙 += vb * vb
        return 点 / (math.sqrt(模甲) * math.sqrt(模乙) + 1e-9)

    相甲 = 全轮状态[0]
    相乙 = 全轮状态[1]
    相丙 = 全轮状态[2]
    相异 = 异状态

    相_甲乙 = 序列相似(相甲, 相乙)
    相_甲丙 = 序列相似(相甲, 相丙)
    相_乙丙 = 序列相似(相乙, 相丙)
    相_基异 = 序列相似(相甲, 相异)

    def 作者相似(序列):
        return sum(签["相似度"] for 签 in 序列) / max(1, len(序列))

    轮均相似 = [作者相似(rs) for rs in 全轮状态]
    总均相似 = sum(轮均相似) / max(1, len(轮均相似))
    方差 = sum((s - 总均相似) ** 2 for s in 轮均相似) / max(1, len(轮均相似))
    均方差 = math.sqrt(方差)

    # ───────────── 报告 ─────────────
    报告路径.write_text(
        "轮\t作者均值相似\t相似均方差\t"
        "甲乙\t甲丙\t乙丙\t基异\t"
        "基社交热\t末社交热\t"
        "帧数\t轮数\n"
    )
    with 报告路径.open("a") as 文:
        文.write(
            f"全\t"
            f"{总均相似:.4f}\t{均方差:.4f}\t"
            f"{相_甲乙:.4f}\t{相_甲丙:.4f}\t{相_乙丙:.4f}\t{相_基异:.4f}\t"
            f"{全轮状态[0][0]['社交热']:.4f}\t"
            f"{全轮状态[2][-1]['社交热']:.4f}\t"
            f"{len(签名集)}\t3\n"
        )

    print("=== 识脸结果 ===")
    print(f"提取签名帧数: {len(签名集)}")
    print(f"作者均值签名: {[round(v,4) for v in 均值签名]}")
    print()
    print(f"  首轮 均值相似: {轮均相似[0]:.4f}")
    print(f"  二轮 均值相似: {轮均相似[1]:.4f}")
    print(f"  三轮 均值相似: {轮均相似[2]:.4f}")
    print(f"  三轮间 均方差: {均方差:.4f}")
    print()
    print(f"  S(轮1,轮2) = {相_甲乙:.4f}")
    print(f"  S(轮1,轮3) = {相_甲丙:.4f}")
    print(f"  S(轮2,轮3) = {相_乙丙:.4f}")
    print(f"  S(同人_vs_异脸) = {相_基异:.4f}")
    print()

    稳度 = (相_甲乙 + 相_甲丙 + 相_乙丙) / 3.0
    稳异差 = 稳度 - 相_基异
    print(f"  稳度均值 S(同人内部): {稳度:.4f}")
    print(f"  稳异差 S(同人)-S(异脸): {稳异差:.4f}")
    print()
    评分 = max(0.0, min(1.0, 稳度 * 0.5 + max(0.0, 稳异差) * 0.5))
    print(f"  稳度评分 [0,1]: {评分:.4f}")
    print(f"主输出文件: {报告路径}")


if __name__ == "__main__":
    跑()
