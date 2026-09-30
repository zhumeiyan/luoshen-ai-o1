"""干净对照实验：每字独立引擎，种子=1。只读数，不判断。"""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

痕文件 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"
if os.path.exists(痕文件):
    os.remove(痕文件)


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
    return np.array(img) / 255.0


def 拼音转听觉(字):
    拼音表 = {
        "我": ("w", "o", 3), "你": ("n", "i", 3), "好": ("h", "ao", 3),
        "来": ("l", "ai", 2), "去": ("q", "u", 4), "走": ("z", "ou", 3),
        "吃": ("ch", "i", 1), "看": ("k", "an", 4), "听": ("t", "ing", 1),
        "说": ("sh", "uo", 1),
    }
    声母能量 = {
        "w": [0.2, 0.1, 0.0, 0.0], "n": [0.3, 0.4, 0.1, 0.0],
        "h": [0.1, 0.3, 0.5, 0.2], "l": [0.4, 0.5, 0.2, 0.1],
        "q": [0.2, 0.6, 0.3, 0.1], "z": [0.5, 0.7, 0.2, 0.0],
        "ch": [0.4, 0.8, 0.3, 0.1], "k": [0.2, 0.5, 0.6, 0.3],
        "t": [0.5, 0.6, 0.2, 0.1], "sh": [0.3, 0.7, 0.4, 0.1],
    }
    韵母重心 = {
        "o": [0.3, 0.5, 0.3, 0.1], "i": [0.1, 0.2, 0.6, 0.8],
        "ao": [0.4, 0.6, 0.4, 0.2], "ai": [0.5, 0.7, 0.5, 0.3],
        "u": [0.2, 0.3, 0.5, 0.4], "ou": [0.3, 0.5, 0.4, 0.2],
        "an": [0.5, 0.6, 0.5, 0.4], "ing": [0.2, 0.4, 0.7, 0.8],
        "uo": [0.3, 0.4, 0.4, 0.3],
    }
    声母, 韵母, 声调 = 拼音表[字]
    起始 = 声母能量.get(声母, [0.25, 0.25, 0.25, 0.25])
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
    return matrix


def 标量(矩阵):
    return float(np.mean(矩阵))


def 数共火(文件路径):
    if not os.path.exists(文件路径):
        return 0
    总 = 0
    with open(文件路径, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                总 += d.get("共火", {}).get("对数", 0)
            except Exception:
                pass
    return 总


# === 主流程 ===
字表 = ["我", "你", "好", "来", "去", "走", "吃", "看", "听", "说"]

# 预渲染字形/字音
字形库 = {}
字音库 = {}
for 字 in 字表:
    字形库[字] = 渲染字形(字)
    字音库[字] = 拼音转听觉(字)

# === 第一组：每字独立引擎，喂 10 次 × 100 拍 ===
print("=" * 60)
print("第一组：每字独立引擎，各喂 10 次 × 100 拍，共 1000 拍")
print("=" * 60)

for 字 in 字表:
    if os.path.exists(痕文件):
        os.remove(痕文件)

    引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
    初筋 = len(引擎.筋树)

    光强 = 标量(字形库[字]) * 600
    声强 = 标量(字音库[字]) * 80

    for 次 in range(10):
        引擎.环境["光"] = 光强 + 引擎.随机器.uniform(-1, 1)
        引擎.环境["声"] = 声强 + 引擎.随机器.uniform(-1, 1)
        引擎.环境["力"] = 10.0
        引擎.环境["温"] = 295.0
        for _ in range(100):
            引擎.推进()

    末筋 = len(引擎.筋树)
    净增 = 末筋 - 初筋
    共火总 = 数共火(痕文件)

    print(f"字'{字}': 初筋{初筋}, 末筋{末筋}, 净增{净增}, 共火{共火总}")

print()
print("注：死筋触发条件 身体稳态.py:313 要求 末传距今>5000 且整千。")
print("第一组每引擎只跑 1000 拍，不触发死筋。")
print("第二组每引擎只跑 1000 拍（500+500），不触发死筋。")

# === 第二组：成对独立引擎，甲单独500拍 + 乙单独500拍，不交替 ===
print()
print("=" * 60)
print("第二组：成对独立引擎，甲 500 拍 + 乙 500 拍，不交替")
print("=" * 60)

字对 = [("我", "你"), ("好", "来"), ("去", "走"), ("吃", "看"), ("听", "说")]

for (字甲, 字乙) in 字对:
    if os.path.exists(痕文件):
        os.remove(痕文件)

    引擎对 = 身体(节点数=96, 脑数=4, 脑间连接数=150, 种子=1)
    初筋数 = len(引擎对.筋树)

    # 甲单独 500 拍（5次×100）
    光甲 = 标量(字形库[字甲]) * 600
    声甲 = 标量(字音库[字甲]) * 80
    for _ in range(5):
        引擎对.环境["光"] = 光甲 + 引擎对.随机器.uniform(-1, 1)
        引擎对.环境["声"] = 声甲 + 引擎对.随机器.uniform(-1, 1)
        引擎对.环境["力"] = 10.0
        引擎对.环境["温"] = 295.0
        for __ in range(100):
            引擎对.推进()

    中筋数 = len(引擎对.筋树)

    # 乙单独 500 拍
    光乙 = 标量(字形库[字乙]) * 600
    声乙 = 标量(字音库[字乙]) * 80
    for _ in range(5):
        引擎对.环境["光"] = 光乙 + 引擎对.随机器.uniform(-1, 1)
        引擎对.环境["声"] = 声乙 + 引擎对.随机器.uniform(-1, 1)
        引擎对.环境["力"] = 10.0
        引擎对.环境["温"] = 295.0
        for __ in range(100):
            引擎对.推进()

    末筋数 = len(引擎对.筋树)
    新增 = 末筋数 - 初筋数
    共火总 = 数共火(痕文件)

    print(f"({字甲}-{字乙}): 初筋{初筋数}, 中甲后{中筋数}, 末筋{末筋数}, 净增{新增}, 共火{共火总}")
