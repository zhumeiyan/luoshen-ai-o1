"""十一个字喂食实验。只读数，不判断。
字形16×16 → 光通道标量；字音16×4 → 声通道标量。
每个字喂10次，报共火、字间连接、矩阵尺寸。
"""
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"


def 取文件大小():
    try:
        return os.path.getsize(痕文件)
    except OSError:
        return 0


def 数共火(起始偏移, 结束偏移):
    """读取痕文件 [起始偏移, 结束偏移) 范围内的共火对数总和。"""
    总 = 0
    if 起始偏移 >= 结束偏移:
        return 0
    with open(痕文件, "r", encoding="utf-8") as f:
        f.seek(起始偏移)
        while f.tell() < 结束偏移:
            line = f.readline()
            if not line:
                break
            try:
                d = json.loads(line)
                总 += d.get("共火", {}).get("对数", 0)
            except Exception:
                pass
    return 总


# === 1. 字形渲染：16×16 矩阵 ===
def 渲染字形(字, 尺寸=16):
    img = Image.new("L", (尺寸, 尺寸), 0)
    draw = ImageDraw.Draw(img)
    字体 = None
    for p in [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]:
        try:
            字体 = ImageFont.truetype(p, 尺寸 - 2)
            break
        except Exception:
            continue
    if 字体 is None:
        字体 = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), 字, font=字体)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (尺寸 - w) // 2 - bbox[0]
    y = (尺寸 - h) // 2 - bbox[1]
    draw.text((x, y), 字, fill=255, font=字体)
    arr = np.array(img) / 255.0
    return arr.tolist(), arr.shape


# === 2. 字音转听觉矩阵：16×4 ===
def 拼音转听觉(字):
    拼音表 = {
        "我": ("w", "o", 3),
        "你": ("n", "i", 3),
        "好": ("h", "ao", 3),
        "来": ("l", "ai", 2),
        "去": ("q", "u", 4),
        "走": ("z", "ou", 3),
        "吃": ("ch", "i", 1),
        "看": ("k", "an", 4),
        "听": ("t", "ing", 1),
        "说": ("sh", "uo", 1),
    }
    声母, 韵母, 声调 = 拼音表[字]

    声母能量 = {
        "w": [0.2, 0.1, 0.0, 0.0], "n": [0.3, 0.4, 0.1, 0.0],
        "h": [0.1, 0.3, 0.5, 0.2], "l": [0.4, 0.5, 0.2, 0.1],
        "q": [0.2, 0.6, 0.3, 0.1], "z": [0.5, 0.7, 0.2, 0.0],
        "ch": [0.4, 0.8, 0.3, 0.1], "k": [0.2, 0.5, 0.6, 0.3],
        "t": [0.5, 0.6, 0.2, 0.1], "sh": [0.3, 0.7, 0.4, 0.1],
    }
    起始 = 声母能量.get(声母, [0.25, 0.25, 0.25, 0.25])

    韵母重心 = {
        "o": [0.3, 0.5, 0.3, 0.1], "i": [0.1, 0.2, 0.6, 0.8],
        "ao": [0.4, 0.6, 0.4, 0.2], "ai": [0.5, 0.7, 0.5, 0.3],
        "u": [0.2, 0.3, 0.5, 0.4], "ou": [0.3, 0.5, 0.4, 0.2],
        "an": [0.5, 0.6, 0.5, 0.4], "ing": [0.2, 0.4, 0.7, 0.8],
        "uo": [0.3, 0.4, 0.4, 0.3],
    }
    重 = 韵母重心.get(韵母, [0.25, 0.25, 0.25, 0.25])

    def 声调轮廓(t, 调):
        if 调 == 1: return 0.8
        if 调 == 2: return 0.3 + 0.5 * t
        if 调 == 3: return 0.6 - 0.3 * np.sin(np.pi * t)
        return 0.9 - 0.6 * t

    matrix = np.zeros((16, 4))
    for t in range(16):
        t_norm = t / 15.0
        基频 = 声调轮廓(t_norm, 声调)
        包络 = np.exp(-t_norm * 1.5)
        for f in range(4):
            matrix[t, f] = (起始[f] * 0.4 + 重[f] * 0.4 + 基频 * 0.2) * 包络
    return matrix.tolist(), matrix.shape


def 标量(矩阵):
    return float(np.mean(矩阵))


# === 初始化痕文件（清空旧数据）===
if os.path.exists(痕文件):
    os.remove(痕文件)

# === 主流程 ===
字表 = ["我", "你", "好", "来", "去", "走", "吃", "看", "听", "说"]
喂食次数 = 10

# --- 矩阵统计 ---
print("=" * 60)
print("一、字形/字音矩阵统计")
print("=" * 60)

字形库 = {}
字音库 = {}
for 字 in 字表:
    形, 形尺寸 = 渲染字形(字)
    音, 音尺寸 = 拼音转听觉(字)
    字形库[字] = 形
    字音库[字] = 音
    print(f"字'{字}': 字形{形尺寸}, 字音{音尺寸}")

# --- 单字喂食：共火 ---
print("\n" + "=" * 60)
print("二、单字喂食 10 次 × 100 拍，共火计数")
print("=" * 60)

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)

共火结果 = {}
for 字 in 字表:
    形矩阵 = 字形库[字]
    音矩阵 = 字音库[字]
    光强 = 标量(形矩阵) * 600
    声强 = 标量(音矩阵) * 80

    引擎.脑共火计数 = [0, 0, 0, 0]
    初筋 = len(引擎.筋树)
    起始偏移 = 取文件大小()

    for 次 in range(喂食次数):
        引擎.环境["光"] = 光强 + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = 声强 + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        for _ in range(100):
            引擎.推进()

    结束偏移 = 取文件大小()
    共火 = 数共火(起始偏移, 结束偏移)
    末筋 = len(引擎.筋树)
    共火结果[字] = 共火
    print(f"字'{字}': 共火 {共火} 对 | 新增筋 {末筋 - 初筋} 根")

# --- 字间连接 ---
print("\n" + "=" * 60)
print("三、字间连接：成对交替喂 (甲3次+乙3次)×100拍")
print("=" * 60)

字对 = [("我", "你"), ("好", "来"), ("去", "走"), ("吃", "看"), ("听", "说")]

# 清空痕文件重来
os.remove(痕文件)

for (字甲, 字乙) in 字对:
    引擎对 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
    初筋数 = len(引擎对.筋树)
    for _ in range(3):
        for 字 in [字甲, 字乙]:
            光 = 标量(字形库[字]) * 600 + 引擎对.随机器.uniform(-1, 1)
            声 = 标量(字音库[字]) * 80 + 引擎对.随机器.uniform(-1, 1)
            引擎对.环境["光"] = 光
            引擎对.环境["声"] = 声
            引擎对.环境["力"] = 10.0
            引擎对.环境["温"] = 295.0
            for __ in range(100):
                引擎对.推进()
    末筋数 = len(引擎对.筋树)
    新增 = 末筋数 - 初筋数
    print(f"({字甲}-{字乙}): 初筋{初筋数} → 末筋{末筋数}, 新增{新增} 根")

# --- 最终汇总 ---
print("\n" + "=" * 60)
print("最终报告")
print("=" * 60)
print(f"\n① 每个字共火次数:")
for 字 in 字表:
    print(f"   '{字}': {共火结果[字]}")
print(f"\n② 矩阵尺寸:")
print(f"   字形: 16×16 (256 像素)")
print(f"   字音: 16×4  (16帧×4频带)")
