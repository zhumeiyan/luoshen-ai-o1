import sys, math, random
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

b = 身体(节点数=96, 种子=1, 上限满值=6.0, 脑数=4)
N = b.节点数
脑数 = b.脑数
脑大小 = b.脑大小
脑名 = ["光","声","力","温"]

# 正弦驱动
光幅=20.0; 光偏=50.0; 光周=2000.0
声幅=15.0; 声偏=45.0; 声周=1500.0
力偏=10.0; 力脉冲概率=0.02; 力脉冲幅=25.0
温偏=295.0; 温漂移幅=3.0

# 延迟分布统计
延迟统计 = {}
for 筋 in b.筋树.values():
    d = 筋['延迟']
    延迟统计[d] = 延迟统计.get(d, 0) + 1
print("延迟分布:", dict(sorted(延迟统计.items())))

def 三元环数():
    邻 = [set() for _ in range(N)]
    for (j, i) in b.筋树:
        if j != i: 邻[j].add(i)
    数 = 0
    for a in range(N):
        for b2 in sorted(邻[a]):
            if b2 <= a: continue
            for c in sorted(邻[b2]):
                if c <= b2: continue
                if a in 邻[c]: 数 += 1
    return 数

def 最长环():
    可达 = [[float('inf')] * N for _ in range(N)]
    for (j, i) in b.筋树:
        if j != i: 可达[j][i] = 1
    for k in range(N):
        for i in range(N):
            if 可达[i][k] == float('inf'): continue
            for j in range(N):
                if 可达[k][j] == float('inf'): continue
                if 可达[i][k] + 可达[k][j] < 可达[i][j]: 可达[i][j] = 可达[i][k] + 可达[k][j]
    最长 = 0
    for i in range(N):
        if 可达[i][i] < float('inf'): 最长 = max(最长, int(可达[i][i]))
    return 最长

print("t(万)  筋数  三元   最长环  光脑范围      声脑范围      力脑范围      温脑范围      上限")

for t in range(1, 200001):
    b.环境["光"] = max(1.0, 光偏 + 光幅 * math.sin(2 * math.pi * t / 光周))
    b.环境["声"] = max(25.0, 声偏 + 声幅 * math.sin(2 * math.pi * t / 声周 + 1.0))
    if random.random() < 力脉冲概率:
        b.环境["力"] = 力偏 + 力脉冲幅 * random.uniform(0.5, 1.5)
    else:
        b.环境["力"] = 力偏 + random.uniform(-1, 1)
    温偏 += random.uniform(-0.1, 0.1) * 温漂移幅
    温偏 = max(288, min(302, 温偏))
    b.环境["温"] = 温偏

    b.推进()

    if t % 50000 == 0:
        筋数 = len(b.筋树)
        三元 = 三元环数()
        最长 = 最长环()
        脑范围 = []
        for nb in range(脑数):
            起点, 终点 = nb * 脑大小, (nb+1) * 脑大小
            值 = [b.激活[i] for i in range(起点, 终点)]
            脑范围.append(f"[{min(值):+.2f},{max(值):+.2f}]")
        print(f"{t//10000:>3}万  {筋数:>4}  {三元:>6}  {最长:>4}    {脑范围[0]:>12} {脑范围[1]:>12} {脑范围[2]:>12} {脑范围[3]:>12} {b.当前上限:.3f}")
