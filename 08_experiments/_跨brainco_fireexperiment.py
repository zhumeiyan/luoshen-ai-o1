"""跨脑共火规则实验: 脑内阈值0.01/概率0.15; 跨脑双节点>0.02/概率0.05. 跑20万拍."""
import time, sys
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 身体稳态 import 身体

引擎 = 身体(节点数=96, 脑数=4, 脑间连接数=150)
脑名 = ["光", "声", "力", "温"]
脑大小 = 引擎.脑大小

# 结构快照函数
def 结构快照(引擎):
    光筋 = 0; 声筋 = 0; 力筋 = 0; 温筋 = 0; 脑间 = 0
    for (j, i) in 引擎.筋树:
        j脑 = j // 脑大小; i脑 = i // 脑大小
        if j脑 == i脑:
            if j脑 == 0: 光筋 += 1
            elif j脑 == 1: 声筋 += 1
            elif j脑 == 2: 力筋 += 1
            else: 温筋 += 1
        else:
            脑间 += 1
    return 光筋, 声筋, 力筋, 温筋, 脑间

# 三元环检测 (Tarjan-based SCC + counting)
def 数三元组(引擎, N):
    from collections import defaultdict
    邻接 = defaultdict(list)
    for (j, i) in 引擎.筋树:
        邻接[j].append(i)
    访问 = [0] * N
    栈 = []
    在栈 = [False] * N
    索引 = [0] * N
    低链 = [0] * N
    索引计数 = [0]
    SCCs = []
    def 强连通(v):
        索引[v] = 低链[v] = 索引计数[0]
        索引计数[0] += 1
        栈.append(v)
        在栈[v] = True
        for w in 邻接[v]:
            if 访问[w] == 0:
                访问[w] = 1
                强连通(w)
                低链[v] = min(低链[v], 低链[w])
            elif 在栈[w]:
                低链[v] = min(低链[v], 索引[w])
        if 低链[v] == 索引[v]:
            comp = []
            while True:
                w = 栈.pop()
                在栈[w] = False
                comp.append(w)
                if w == v:
                    break
            SCCs.append(comp)
    for v in range(N):
        if 访问[v] == 0:
            访问[v] = 1
            强连通(v)
    三元 = 0
    for comp in SCCs:
        if len(comp) == 3:
            三元 += 1
    return 三元

# Floyd 最长环
def 最长环(N, 筋树):
    INF = 1e9
    D = [[INF]*N for _ in range(N)]
    for (j, i) in 筋树:
        D[j][i] = 1
    for k in range(N):
        for i in range(N):
            if D[i][k] == INF: continue
            for j in range(N):
                if D[k][j] == INF: continue
                if D[i][j] > D[i][k] + D[k][j]:
                    D[i][j] = D[i][k] + D[k][j]
    最长 = 0
    for i in range(N):
        if D[i][i] < INF and D[i][i] > 最长:
            最长 = D[i][i]
    return 最长

# 上限估计 (Floyd-Warshall)
def 上限估计(N, 筋树):
    INF = 1e9
    D = [[INF]*N for _ in range(N)]
    for (j, i) in 筋树:
        D[j][i] = 1
    for i in range(N):
        D[i][i] = 0
    for k in range(N):
        for i in range(N):
            if D[i][k] == INF: continue
            for j in range(N):
                if D[k][j] == INF: continue
                if D[i][j] > D[i][k] + D[k][j]:
                    D[i][j] = D[i][k] + D[k][j]
    所有路径 = []
    for i in range(N):
        for j in range(N):
            if D[i][j] < INF:
                所有路径.append(D[i][j])
    if not 所有路径:
        return 0.0
    return max(所有路径)

# 光脑范围
def 脑范围(引擎, 脑号):
    起点 = 脑号 * 脑大小
    终点 = 起点 + 脑大小
    激活值 = []
    for i in range(起点, 终点):
        激活值.append(引擎._激活[i] if hasattr(引擎, '_激活') else 引擎.激活[i])
    if not 激活值:
        return "[0,0]"
    return f"[{min(激活值):+.3f},{max(激活值):+.3f}]"

# 运行
print(f"初始筋树={len(引擎.筋树)}")
print(f"t(万)  光  声  力  温  脑间  总筋  三元  最长环  上限  光脑范围      声脑范围      力脑范围      温脑范围      激活标准差(光声力温)")

总拍数 = 200000
报告间隔 = 50000
下一报告 = 报告间隔
开始 = time.time()

# 记录脑间连接趋势
脑间历史 = []

for t in range(1, 总拍数 + 1):
    引擎.推进()
    if t == 下一报告:
        光, 声, 力, 温, 脑间 = 结构快照(引擎)
        总筋 = 光 + 声 + 力 + 温 + 脑间
        三元 = 数三元组(引擎, 引擎.节点数)
        最长 = 最长环(引擎.节点数, 引擎.筋树)
        上限 = 上限估计(引擎.节点数, 引擎.筋树)
        脑间历史.append((t, 脑间))
        # 激活标准差
        import numpy as np
        act = np.array(引擎._激活 if hasattr(引擎, '_激活') else 引擎.激活)
        stds = []
        for nb in range(引擎.脑数):
            s = act[nb*脑大小:(nb+1)*脑大小].std()
            stds.append(s)
        # 范围
        范围 = [脑范围(引擎, nb) for nb in range(引擎.脑数)]
        print(f"{t//10000:4d}万  {光:3d} {声:3d} {力:3d} {温:3d}  {脑间:4d}  {总筋:5d}  {三元:5d}  {最长:5d}  {上限:.3f}  {范围[0]}  {范围[1]}  {范围[2]}  {范围[3]}  {stds[0]:.3f} {stds[1]:.3f} {stds[2]:.3f} {stds[3]:.3f}")
        共火 = getattr(引擎, '脑共火计数', [0,0,0,0])
        print(f"      共火: 光={共火[0]} 声={共火[1]} 力={共火[2]} 温={共火[3]}")
        下一报告 += 报告间隔

耗时 = time.time() - 开始

# 趋势分析
print(f"\n脑间连接趋势:")
for t, n in 脑间历史:
    print(f"  {t//10000}万拍: {n}")
if len(脑间历史) >= 2:
    增长 = 脑间历史[-1][1] - 脑间历史[0][1]
    print(f"  5万→20万: {'上涨' if 增长 > 0 else '下降'} {abs(增长)}")
else:
    print("  数据不足")

print(f"\n耗时: {耗时:.1f}s ({总拍数/耗时:.0f} 拍/秒)")
