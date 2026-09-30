"""回接实验: 5维感受器(含回接差通道), 20节点回接."""
import sys, time
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体
import numpy as np
from collections import defaultdict

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150)
脑大小 = 引擎.脑大小
回接节点 = 引擎._回接节点
回接集 = set(回接节点)

def 结构快照():
    脑内筋 = [0]*4
    脑间 = 0
    回接自指 = 0
    for (j, i) in 引擎.筋树:
        j脑, i脑 = j//脑大小, i//脑大小
        if j脑 == i脑:
            脑内筋[j脑] += 1
        else:
            脑间 += 1
        if j in 回接集 and i in 回接集:
            回接自指 += 1
    return 脑内筋[0], 脑内筋[1], 脑内筋[2], 脑内筋[3], 脑间, 回接自指

def 数三元组(N):
    邻接 = defaultdict(list)
    for (j, i) in 引擎.筋树:
        邻接[j].append(i)
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
    for (j, i) in 引擎.筋树:
        D[j][i] = 1
    for k in range(N):
        for i in range(N):
            if D[i][k] == INF: continue
            for j in range(N):
                if D[k][j] == INF: continue
                if D[i][j] > D[i][k]+D[k][j]:
                    D[i][j] = D[i][k]+D[k][j]
    最长 = 0
    for i in range(N):
        if D[i][i] < INF and D[i][i] > 最长:
            最长 = D[i][i]
    return 最长

def 上限估计(N):
    INF = 1e9
    D = [[INF]*N for _ in range(N)]
    for (j, i) in 引擎.筋树:
        D[j][i] = 1
    for i in range(N):
        D[i][i] = 0
    for k in range(N):
        for i in range(N):
            if D[i][k] == INF: continue
            for j in range(N):
                if D[k][j] == INF: continue
                if D[i][j] > D[i][k]+D[k][j]:
                    D[i][j] = D[i][k]+D[k][j]
    return max(D[i][j] for i in range(N) for j in range(N) if D[i][j] < INF)

def 脑范围(脑号):
    act = np.array(引擎.激活)
    段 = act[脑号*脑大小:(脑号+1)*脑大小]
    return f"[{min(段):+.3f},{max(段):+.3f}]"

print(f"初始筋树={len(引擎.筋树)}, 回接节点={回接节点}")
print("t(万)  光  声  力  温  脑间  总筋  自指  三元  最长环  上限  光脑范围      声脑范围      力脑范围      温脑范围")

总拍数 = 200000
报告间隔 = 50000
下一报告 = 报告间隔
开始 = time.time()

for t in range(1, 总拍数+1):
    引擎.推进()
    if t == 下一报告:
        光, 声, 力, 温, 脑间, 自指 = 结构快照()
        总筋 = 光+声+力+温+脑间
        三元 = 数三元组(引擎.节点数)
        最长 = 最长路径(引擎.节点数)
        上限 = 上限估计(引擎.节点数)
        范围 = [脑范围(nb) for nb in range(4)]
        print(f"{t//10000:4d}万  {光:3d} {声:3d} {力:3d} {温:3d}  {脑间:4d}  {总筋:5d}  {自指:4d}  {三元:5d}  {最长:5d}  {上限:.3f}  {范围[0]}  {范围[1]}  {范围[2]}  {范围[3]}")
        共火 = 引擎.脑共火计数
        print(f"      共火: 光={共火[0]} 声={共火[1]} 力={共火[2]} 温={共火[3]}")
        下一报告 += 报告间隔

耗时 = time.time() - 开始

# 最终详细分析
print(f"\n=== 回接差分析 ===")
act = np.array(引擎.激活)
回接激活 = [act[j] for j in 回接节点]
print(f"回接节点激活: mean={np.mean(回接激活):.4f}, std={np.std(回接激活):.4f}")
print(f"回接节点在筋树中有传出筋的:")
回接传出 = [(j,i) for (j,i) in 引擎.筋树 if j in 回接集]
print(f"  从回接节点出发的筋: {len(回接传出)}")
回接传入 = [(j,i) for (j,i) in 引擎.筋树 if i in 回接集]
print(f"  进入回接节点的筋: {len(回接传入)}")

# 检测回接是否在自指回路中形成紧密子图
回接内筋 = [(j,i) for (j,i) in 引擎.筋树 if j in 回接集 and i in 回接集]
print(f"回接内筋(回接节点互指): {len(回接内筋)}")
if 回接内筋:
    print("  具体连接:", 回接内筋[:20])

print(f"\n耗时: {耗时:.1f}s ({总拍数/耗时:.0f} 拍/秒)")
