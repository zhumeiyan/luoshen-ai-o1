"""自己点实验: 每1000拍, 自指簇选节点+0.5激活, 下次选反馈差最大的."""
import sys, time
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体
import numpy as np
from collections import defaultdict

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150)
脑大小 = 引擎.脑大小

def 自指簇():
    回接集 = set(引擎._回接节点)
    簇 = set()
    for (j, i) in 引擎.筋树:
        if j in 回接集 and i in 回接集:
            簇.add(j); 簇.add(i)
    return 簇

def 结构快照():
    脑内筋 = [0]*4; 脑间 = 0
    for (j, i) in 引擎.筋树:
        if j//脑大小 == i//脑大小:
            脑内筋[j//脑大小] += 1
        else:
            脑间 += 1
    return 脑内筋[0], 脑内筋[1], 脑内筋[2], 脑内筋[3], 脑间

def 数三元组(N):
    邻接 = defaultdict(list)
    for (j, i) in 引擎.筋树: 邻接[j].append(i)
    访问 = [0]*N; 栈 = []; 在栈 = [False]*N
    索引 = [0]*N; 低链 = [0]*N; 索引计数 = [0]; SCCs = []
    def 强连通(v):
        索引[v] = 低链[v] = 索引计数[0]; 索引计数[0] += 1
        栈.append(v); 在栈[v] = True
        for w in 邻接[v]:
            if 访问[w] == 0:
                访问[w] = 1; 强连通(w)
                低链[v] = min(低链[v], 低链[w])
            elif 在栈[w]:
                低链[v] = min(低链[v], 索引[w])
        if 低链[v] == 索引[v]:
            comp = []
            while True:
                w = 栈.pop(); 在栈[w] = False; comp.append(w)
                if w == v: break
            SCCs.append(comp)
    for v in range(N):
        if 访问[v] == 0:
            访问[v] = 1; 强连通(v)
    return sum(1 for comp in SCCs if len(comp)==3)

def 最长路径(N):
    INF = 1e9
    D = [[INF]*N for _ in range(N)]
    for (j, i) in 引擎.筋树: D[j][i] = 1
    for k in range(N):
        for i in range(N):
            if D[i][k] == INF: continue
            for j in range(N):
                if D[k][j] == INF: continue
                D[i][j] = min(D[i][j], D[i][k]+D[k][j])
    最长 = 0
    for i in range(N):
        if D[i][i] < INF and D[i][i] > 最长: 最长 = D[i][i]
    return 最长

def 上限估计(N):
    INF = 1e9
    D = [[INF]*N for _ in range(N)]
    for (j, i) in 引擎.筋树: D[j][i] = 1
    for i in range(N): D[i][i] = 0
    for k in range(N):
        for i in range(N):
            if D[i][k] == INF: continue
            for j in range(N):
                if D[k][j] == INF: continue
                D[i][j] = min(D[i][j], D[i][k]+D[k][j])
    return max(D[i][j] for i in range(N) for j in range(N) if D[i][j] < INF)

def 脑范围(脑号):
    act = np.array(引擎.激活)
    段 = act[脑号*脑大小:(脑号+1)*脑大小]
    return f"[{min(段):+.3f},{max(段):+.3f}]"

print(f"初始筋树={len(引擎.筋树)}, 回接节点={引擎._回接节点}")
print("t(万)  光  声  力  温  脑间  总筋  自指筋  自指点  三元  最长环  上限  光脑范围")

总拍数 = 200000
报告间隔 = 50000
下一报告 = 报告间隔
开始 = time.time()

for t in range(1, 总拍数+1):
    引擎.推进()
    if t == 下一报告:
        光, 声, 力, 温, 脑间 = 结构快照()
        总筋 = 光+声+力+温+脑间
        簇 = 自指簇()
        自指筋 = sum(1 for (j,i) in 引擎.筋树 if j in 簇 and i in 簇)
        点次数 = getattr(引擎, '_点次数', 0)
        三元 = 数三元组(引擎.节点数)
        最长 = 最长路径(引擎.节点数)
        上限 = 上限估计(引擎.节点数)
        范围 = [脑范围(nb) for nb in range(4)]
        print(f"{t//10000:4d}万  {光:3d} {声:3d} {力:3d} {温:3d}  {脑间:4d}  {总筋:5d}  {自指筋:5d}  {点次数//1000:4d}k  {三元:5d}  {最长:5d}  {上限:.3f}  {范围[0]}")
        共火 = 引擎.脑共火计数
        print(f"      共火: 光={共火[0]} 声={共火[1]} 力={共火[2]} 温={共火[3]}, 自指簇={sorted(簇)}")
        下一报告 += 报告间隔

耗时 = time.time() - 开始

# 五项报告
簇 = 自指簇()
自指筋 = sum(1 for (j,i) in 引擎.筋树 if j in 簇 and i in 簇)
print(f"\n=== 接差到筋树 五项报告 ===")
print(f"\n① 自指筋: {自指筋}")
print(f"② 自指簇: 大小={len(簇)} 节点={sorted(簇)}")

# ③ 位点分布
位点列表 = [筋['位点'] for 筋 in 引擎.筋树.values()]
序列 = getattr(引擎, '_点序列', [])
位点总 = sum(位点列表)
位点均 = 位点总 / len(位点列表)
位点最大 = max(位点列表)
位点最小 = min(位点列表)
# 分布: 位点=1的有多少, >1的有多少
原始数 = sum(1 for x in 位点列表 if x == 1)
增长数 = sum(1 for x in 位点列表 if x > 1)
超粗数 = sum(1 for x in 位点列表 if x >= 10)
print(f"\n③ 位点分布:")
print(f"   筋总数: {len(位点列表)}")
print(f"   位点总和: {位点总}  均值: {位点均:.2f}  最大: {位点最大}  最小: {位点最小}")
print(f"   原始(=1): {原始数}  增长(>1): {增长数}  超粗(≥10): {超粗数}")

# ④ 上限
上限 = 上限估计(引擎.节点数)
print(f"\n④ 上限: {上限:.3f}")

# ⑤ 被点节点相邻筋粗度
# 找出所有被点过的节点
被点节点集 = set(序列)
# 收集这些节点相邻的筋的位点
相邻筋位点 = []
非相邻筋位点 = []
for (j, i), 筋 in 引擎.筋树.items():
    if j in 被点节点集 or i in 被点节点集:
        相邻筋位点.append(筋['位点'])
    else:
        非相邻筋位点.append(筋['位点'])
if 相邻筋位点:
    相邻总 = sum(相邻筋位点)
    相邻均 = 相邻总 / len(相邻筋位点)
    相邻最大 = max(相邻筋位点)
    非相邻均 = sum(非相邻筋位点) / len(非相邻筋位点) if 非相邻筋位点 else 0
    print(f"\n⑤ 被点节点相邻筋粗度:")
    print(f"   相邻筋数: {len(相邻筋位点)}  位点总和: {相邻总}  均值: {相邻均:.2f}  最大: {相邻最大}")
    print(f"   非相邻筋数: {len(非相邻筋位点)}  均值: {非相邻均:.2f}")
    print(f"   相邻/非相邻 均值比: {相邻均/非相邻均:.2f}" if 非相邻均 > 0 else "")

print(f"\n耗时: {耗时:.1f}s ({总拍数/耗时:.0f} 拍/秒)")
print(f"\n对比——无沉淀自己点实验20万: 自指=52, 上限=2.000, 总点=200")
