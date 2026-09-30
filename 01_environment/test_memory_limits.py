"""
test_memory_limits.py — 人类记忆式压力测试
============================================
逐步增加学习负载，探测G矩阵承载上限。

人类记忆特征模拟：
- 工作记忆容量有限（7±2概念）
- 遗忘曲线：先快后慢
- 前摄/倒摄干扰
- 提取依赖线索质量
"""
import numpy as np
import random
import time
from memory_system import HebbianMemory

def generate_experience(n_concepts, cluster_size):
    """生成一次经历：cluster_size个相关概念同时激活"""
    center = random.randint(0, n_concepts - 1)
    concepts = {center}
    for _ in range(cluster_size - 1):
        # 相关概念在邻近ID（模拟语义聚类）
        offset = random.randint(-5, 5)
        c = (center + offset) % n_concepts
        concepts.add(c)
    return frozenset(concepts)

def test_capacity(n_concepts=66, n_experiences=100, cluster_size=3):
    """
    测试记忆容量：学习n_experiences次经历，测量回忆准确率。
    
    每个经历 = cluster_size个概念同时激活（模拟"乌鸦+石子+渴"这种事件）
    """
    mem = HebbianMemory(dim=1024, n_concepts=n_concepts)
    experiences = []
    
    # 学习阶段
    for i in range(n_experiences):
        exp = generate_experience(n_concepts, cluster_size)
        experiences.append(exp)
        mem.experience(exp)
    
    # 测试回忆：对每个经历，给一半线索，看能否想起另一半
    correct = 0
    false_positive = 0
    false_negative = 0
    
    for exp in experiences:
        cue = set(list(exp)[:cluster_size//2])  # 一半线索
        target = exp - cue  # 另一半目标
        
        recalled = mem.recall(cue)
        
        # 正确回忆：目标概念被想起
        hit = len(target & recalled)
        # 错误激活：想起非目标概念
        fp = len(recalled - target - cue)
        # 遗漏：目标没被想起
        fn = len(target - recalled)
        
        if hit == len(target):
            correct += 1
        false_positive += fp
        false_negative += fn
    
    n_tests = len(experiences)
    stats = {
        "experiences": n_experiences,
        "recall_rate": correct / n_tests,
        "false_positive_rate": false_positive / (n_tests * n_concepts),
        "false_negative_rate": false_negative / (n_tests * len(target)),
        "G_nonzero": mem.memory_stats()["G_nonzero"],
        "G_sparsity": mem.memory_stats()["G_sparsity"],
    }
    return stats

def test_forgetting_curve(n_concepts=66, n_experiences=50):
    """
    遗忘曲线测试：学习后间隔不同"时间"（遗忘迭代次数），测量回忆率。
    """
    mem = HebbianMemory(dim=1024, n_concepts=n_concepts)
    experiences = []
    
    for _ in range(n_experiences):
        exp = generate_experience(n_concepts, 3)
        experiences.append(exp)
        mem.experience(exp)
    
    # 不同遗忘间隔
    intervals = [0, 100, 500, 1000, 5000, 10000]
    results = []
    
    for interval in intervals:
        mem_copy = HebbianMemory(dim=1024, n_concepts=n_concepts)
        mem_copy.G = mem.G.copy()
        mem_copy.S = mem.S.copy()
        
        # 模拟时间流逝：遗忘迭代
        for _ in range(interval):
            mem_copy.G *= (1 - mem_copy.FORGETTING_RATE)
        
        # 测试回忆
        correct = 0
        for exp in experiences:
            cue = {list(exp)[0]}
            target = exp - cue
            recalled = mem_copy.recall(cue)
            if target <= recalled:
                correct += 1
        
        results.append({
            "interval": interval,
            "recall_rate": correct / n_experiences,
            "G_nonzero": np.count_nonzero(mem_copy.G > 0),
        })
    
    return results

def test_interference(n_concepts=66, n_old=30, n_new=30):
    """
    干扰测试：先学旧知识，再学新知识，测试前摄/倒摄干扰。
    """
    mem = HebbianMemory(dim=1024, n_concepts=n_concepts)
    
    # 旧知识：概念0-29
    old_experiences = []
    for _ in range(n_old):
        exp = frozenset(random.sample(range(0, 30), 3))
        old_experiences.append(exp)
        mem.experience(exp)
    
    # 新知识：概念30-59（与旧知识不重叠）
    new_experiences = []
    for _ in range(n_new):
        exp = frozenset(random.sample(range(30, 60), 3))
        new_experiences.append(exp)
        mem.experience(exp)
    
    # 测试：旧知识回忆率
    old_correct = 0
    for exp in old_experiences:
        cue = {list(exp)[0]}
        target = exp - cue
        recalled = mem.recall(cue)
        if target <= recalled:
            old_correct += 1
    
    # 测试：新知识回忆率
    new_correct = 0
    for exp in new_experiences:
        cue = {list(exp)[0]}
        target = exp - cue
        recalled = mem.recall(cue)
        if target <= recalled:
            new_correct += 1
    
    return {
        "old_recall": old_correct / n_old,
        "new_recall": new_correct / n_new,
    }

if __name__ == "__main__":
    print("=" * 60)
    print("洛神记忆系统 — 人类记忆式压力测试")
    print("=" * 60)
    
    # 测试1：容量上限
    print("\n=== T1 容量测试（逐步增加经历数量） ===")
    capacities = [10, 50, 100, 200, 500, 1000]
    for cap in capacities:
        stats = test_capacity(n_experiences=cap)
        print(f"  经历{cap:4d}: 回忆率={stats['recall_rate']:.1%}, "
              f"假阳性={stats['false_positive_rate']:.3f}, "
              f"G连接={stats['G_nonzero']:6d}, "
              f"稀疏度={stats['G_sparsity']}")
    
    # 测试2：遗忘曲线
    print("\n=== T2 遗忘曲线（50个经历，不同间隔） ===")
    forgetting = test_forgetting_curve()
    for r in forgetting:
        print(f"  遗忘迭代{r['interval']:6d}: 回忆率={r['recall_rate']:.1%}, "
              f"G连接={r['G_nonzero']}")
    
    # 测试3：干扰
    print("\n=== T3 干扰测试（先学30个旧知识，再学30个新知识） ===")
    interference = test_interference()
    print(f"  旧知识回忆率: {interference['old_recall']:.1%}")
    print(f"  新知识回忆率: {interference['new_recall']:.1%}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
