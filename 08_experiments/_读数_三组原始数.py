"""读数：接差到筋树实验三组原始数。只读数，不判断。"""
import sys, time
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体
from collections import Counter, defaultdict

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150)

总拍数 = 200000
print(f"初始筋树={len(引擎.筋树)}, 回接节点={引擎._回接节点}")
print(f"开始跑 {总拍数} 拍...")
开始 = time.time()

for t in range(1, 总拍数+1):
    引擎.推进()

耗时 = time.time() - 开始
print(f"跑完。耗时 {耗时:.1f}s\n")

# === 自指簇 ===
回接集 = set(引擎._回接节点)
簇 = set()
for (j, i) in 引擎.筋树:
    if j in 回接集 and i in 回接集:
        簇.add(j); 簇.add(i)

# === 第一组：200次点落在哪 ===
序列 = getattr(引擎, '_点序列', [])
计数 = Counter(序列)

print("=" * 60)
print("第一组：200次点，落在哪")
print("=" * 60)
print(f"\n① 200次点，不同节点个数: {len(计数)}")
print(f"\n② 每个节点被点次数:")
for 节点 in sorted(计数.keys()):
    print(f"   节点{节点:3d}: {计数[节点]:3d} 次")

# === 第二组：被点节点的出筋/入筋 ===
print("\n" + "=" * 60)
print("第二组：被点节点的出筋数/入筋数")
print("=" * 60)

# 统计每个节点的出筋和入筋
出筋 = defaultdict(list)
入筋 = defaultdict(list)
for (j, i), 筋 in 引擎.筋树.items():
    出筋[j].append((j, i))
    入筋[i].append((j, i))

print(f"\n①② 每个被点节点的出筋数/入筋数:")
for 节点 in sorted(计数.keys()):
    print(f"   节点{节点:3d}: 出筋={len(出筋[节点]):3d}  入筋={len(入筋[节点]):3d}")

# 相邻筋的定义
被点节点集 = set(序列)  # 所有被点过的不同节点
相邻筋 = []
非相邻筋 = []
for (j, i), 筋 in 引擎.筋树.items():
    if j in 被点节点集 or i in 被点节点集:
        相邻筋.append(((j, i), 筋))
    else:
        非相邻筋.append(((j, i), 筋))

# 拆解：出筋（被点节点为起点）vs 入筋（被点节点为终点）
相邻出筋 = [筋 for (j, i), 筋 in 相邻筋 if j in 被点节点集]
相邻入筋 = [筋 for (j, i), 筋 in 相邻筋 if i in 被点节点集]

print(f"\n③ 相邻筋455的构成:")
print(f"   相邻筋总数: {len(相邻筋)}")
print(f"   其中 被点节点为起点(出筋): {len(相邻出筋)}")
print(f"   其中 被点节点为终点(入筋): {len(相邻入筋)}")
print(f"   出筋+入筋 = {len(相邻出筋)} + {len(相邻入筋)} = {len(相邻出筋)+len(相邻入筋)}")
print(f"   注: 若一条筋 起点∈被点 且 终点∈被点，同时计入出筋和入筋")
# 去重：两端都被点的筋
双端 = [筋 for (j, i), 筋 in 相邻筋 if j in 被点节点集 and i in 被点节点集]
单端出 = [筋 for (j, i), 筋 in 相邻筋 if j in 被点节点集 and i not in 被点节点集]
单端入 = [筋 for (j, i), 筋 in 相邻筋 if j not in 被点节点集 and i in 被点节点集]
print(f"   双端都在被点集的筋: {len(双端)}")
print(f"   仅起点在被点集的筋: {len(单端出)}")
print(f"   仅终点在被点集的筋: {len(单端入)}")
print(f"   双端+单端出+单端入 = {len(双端)}+{len(单端出)}+{len(单端入)} = {len(双端)+len(单端出)+len(单端入)}")

# === 第三组：位点16.04怎么来的 ===
print("\n" + "=" * 60)
print("第三组：位点16.04怎么来的")
print("=" * 60)

# 相邻筋位点分布
相邻位点列表 = [筋['位点'] for _, 筋 in 相邻筋]
print(f"\n① 相邻筋位点总和: {sum(相邻位点列表)}, 数量: {len(相邻位点列表)}, 均值: {sum(相邻位点列表)/len(相邻位点列表):.2f}")

print(f"\n② 相邻筋位点分布:")
位点计数 = Counter(相邻位点列表)
for 位点值 in sorted(位点计数.keys()):
    print(f"   位点={位点值:2d}: {位点计数[位点值]:4d} 根")
print(f"   汇总: 位点=1: {位点计数.get(1,0)}, 位点2-9: {sum(位点计数.get(v,0) for v in range(2,10))}, 位点≥10: {sum(位点计数.get(v,0) for v in range(10,41))}, 位点=40: {位点计数.get(40,0)}")

# 相邻筋传导次数分布
print(f"\n③ 相邻筋传导次数分布:")
相邻传导列表 = [筋['传导次数'] for _, 筋 in 相邻筋]
传导计数 = Counter(相邻传导列表)
for 传导值 in sorted(传导计数.keys()):
    if 传导值 <= 20 or 传导计数[传导值] >= 5:
        print(f"   传导={传导值:5d}: {传导计数[传导值]:4d} 根")
print(f"   最大传导次数: {max(相邻传导列表) if 相邻传导列表 else 0}")

# 相邻筋疲劳分布
print(f"\n   相邻筋疲劳分布:")
相邻疲劳列表 = [round(筋['疲劳'], 2) for _, 筋 in 相邻筋]
疲劳计数 = Counter(相邻疲劳列表)
for 疲劳值 in sorted(疲劳计数.keys()):
    if 疲劳计数[疲劳值] >= 3:
        print(f"   疲劳={疲劳值:.2f}: {疲劳计数[疲劳值]:4d} 根")

# 5%抽签规则原文
print(f"\n④ 5%抽签规则原文:")
print(f"   文件: 身体稳态.py, 行号: 278-296")
print(f"   行278-282:")
print(f"     if 筋['传导次数'] < 3: continue")
print(f"     if 筋['疲劳'] < 0.5: continue")
print(f"     活跃 = self.激活[j] > 0 and self.激活[i] > 0")
print(f"     if not 活跃: continue")
print(f"   行294-296:")
print(f"     if 筋['疲劳'] > 0.5 and random.random() < 0.05:")
print(f"         筋['位点'] = min(40, 筋['位点'] + 1)")

# 触发放大: 被点哪些节点被点后激活最高峰多少
print(f"\n" + "=" * 60)
print("附：点序列完整记录")
print("=" * 60)
print(f"序列长度: {len(序列)}")
for i in range(0, len(序列), 50):
    print(f"  [{i:3d}-{min(i+50,len(序列))-1:3d}]: {序列[i:i+50]}")

# 写入原始数据到文件
with open("/mnt/data/catpaw/home/workspace/洛神赋1.1/_原始数_三组.txt", "w", encoding="utf-8") as f:
    f.write("接差到筋树实验 三组原始数\n")
    f.write(f"总拍数: {总拍数}, 耗时: {耗时:.1f}s\n\n")
    f.write("第一组: 200次点落在哪\n")
    f.write(f"不同节点个数: {len(计数)}\n")
    for 节点 in sorted(计数.keys()):
        f.write(f"  节点{节点}: {计数[节点]} 次\n")
    f.write(f"\n第二组: 被点节点出筋/入筋\n")
    for 节点 in sorted(计数.keys()):
        f.write(f"  节点{节点}: 出筋={len(出筋[节点])} 入筋={len(入筋[节点])}\n")
    f.write(f"\n相邻筋总数: {len(相邻筋)}\n")
    f.write(f"出筋(被点起点): {len(相邻出筋)}\n")
    f.write(f"入筋(被点终点): {len(相邻入筋)}\n")
    f.write(f"\n第三组: 位点分布\n")
    for 位点值 in sorted(位点计数.keys()):
        f.write(f"  位点={位点值}: {位点计数[位点值]} 根\n")

print(f"\n原始数据已写入: /mnt/data/catpaw/home/workspace/洛神赋1.1/_原始数_三组.txt")
