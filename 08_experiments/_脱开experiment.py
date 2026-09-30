"""脱开实验: 同节点连点记录×0.9, 换节点恢复1.0, 记录<0.3不可选."""
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
    adj = defaultdict(list)
    for (j, i) in 引擎.筋树: adj[j].append(i)
    visit = [0]*N; stack = []; onstack = [False]*N
    idx = [0]*N; low = [0]*N; ic = [0]; sccs = []
    def sc(v):
        idx[v] = low[v] = ic[0]; ic[0] += 1
        stack.append(v); onstack[v] = True
        for w in adj[v]:
            if visit[w] == 0:
                visit[w] = 1; sc(w); low[v] = min(low[v], low[w])
            elif onstack[w]:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop(); onstack[w] = False; comp.append(w)
                if w == v: break
            sccs.append(comp)
    for v in range(N):
        if visit[v] == 0:
            visit[v] = 1; sc(v)
    return sum(1 for c in sccs if len(c)==3)

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
    return max((D[i][i] for i in range(N) if D[i][i] < INF), default=0)

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

# 跟踪每节点被点次数和记录
节点点次 = defaultdict(int)

print(f"初始筋树={len(引擎.筋树)}, 回接节点={引擎._回接节点}")
print("t(万)  光  声  力  温  脑间  总筋  自指筋  自指点  三元  最长环  上限")

总拍数 = 200000
报告间隔 = 50000
下一报告 = 报告间隔
开始 = time.time()

# 包装原始的_自己点以统计节点
原方法 = 引擎._自己点
def 自己点包装(记录=None):
    原方法(记录)
    if hasattr(引擎, '_上次点目标'):
        节点点次[引擎._上次点目标] += 1
引擎._自己点 = 自己点包装

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
        print(f"{t//10000:4d}万  {光:3d} {声:3d} {力:3d} {温:3d}  {脑间:4d}  {总筋:5d}  {自指筋:5d}  {点次数//1000:4d}k  {三元:5d}  {最长:5d}  {上限:.3f}")
        共火 = 引擎.脑共火计数
        print(f"      共火: 光={共火[0]} 声={共火[1]} 力={共火[2]} 温={共火[3]}, 簇={sorted(簇)}")
        点记录 = getattr(引擎, '_点记录', {})
        if 点记录:
            记录显示 = {k: f"{v:.2f}" for k, v in sorted(点记录.items(), key=lambda x: x[1])}
            print(f"      记录(低→高): {记录显示}")
        下一报告 += 报告间隔

耗时 = time.time() - 开始
簇 = 自指簇()
自指筋 = sum(1 for (j,i) in 引擎.筋树 if j in 簇 and i in 簇)

print(f"\n=== 脱开分析 ===")
print(f"自指筋: {自指筋}")
print(f"自指簇(节点): {sorted(簇)} ({len(簇)}个)")
print(f"总点数: {getattr(引擎, '_点次数', 0)}")
print(f"节点点次:")
for n in sorted(节点点次.keys()):
    print(f"  节点{n}: {节点点次[n]}次")
换节点数 = len(节点点次)
print(f"换过几个节点: {换节点数}")
记录 = getattr(引擎, '_点记录', {})
print(f"最终记录: {dict(sorted(记录.items()))}")

print(f"\n对比——不脱开(自己点实验)20万:")
print(f"  自指筋=42, 簇14节点, 34号独大, 上限=3.000")

print(f"\n耗时: {耗时:.1f}s ({总拍数/耗时:.0f} 拍/秒)")
