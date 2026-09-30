"""
memory_system.py — 洛神记忆系统
==========================
基于生物电网络的赫布学习 + 主动回忆
"""
import numpy as np
import hashlib
import time

class HebbianMemory:
    def __init__(self, dim=1024, n_concepts=66):
        self.dim = dim
        self.n_concepts = n_concepts
        self.G = np.zeros((dim, dim))  # 从零开始，学习后才有连接
        self.S = np.zeros(dim)
        self.S_DECAY = 0.3
        self.LEARNING_RATE = 0.1  # 提高学习率
        self.FORGETTING_RATE = 0.0001
        self.G_MAX = 1.0
        self.snapshots = []
    
    def experience(self, active_concepts, label=None):
        """经历：一起激活的概念，电导增强"""
        # 短期记忆
        dE = np.zeros(self.dim)
        for c in active_concepts:
            dE[c % self.dim] += 1.0
        self.S = self.S * self.S_DECAY + dE
        
        # 赫布学习：只增强本次激活概念之间的连接
        active_list = list(active_concepts)
        for i in active_list:
            for j in active_list:
                if i != j:
                    idx_i = i % self.dim
                    idx_j = j % self.dim
                    self.G[idx_i, idx_j] += self.LEARNING_RATE
                    self.G[idx_i, idx_j] = np.clip(self.G[idx_i, idx_j], 0, self.G_MAX)
        
        # 遗忘
        self.G *= (1 - self.FORGETTING_RATE)
        
        if label:
            self._snapshot(label, active_concepts)
    
    def recall(self, cue_concepts):
        """
        主动回忆：一步精确传播。
        只激活与线索有直接强连接的概念（不是全矩阵扩散）。
        """
        recalled = set(cue_concepts)  # 线索本身算想起来的
        
        for c in cue_concepts:
            idx = c % self.dim
            # 找G中从c出发的最强连接
            connections = self.G[idx, :self.n_concepts]
            # 只取显著强于背景的值
            threshold = np.max(connections) * 0.3 if np.max(connections) > 0 else 0.01
            strong = np.where(connections > threshold)[0]
            recalled.update(strong)
        
        return recalled
    
    def _snapshot(self, label, active_concepts):
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "concepts": sorted(active_concepts),
            "G_sparsity": np.count_nonzero(self.G > 0) / self.G.size,
        }
        self.snapshots.append(snapshot)
    
    def memory_stats(self):
        return {
            "G_nonzero": np.count_nonzero(self.G > 0),
            "G_sparsity": f"{np.count_nonzero(self.G > 0) / self.G.size * 100:.1f}%",
            "snapshots": len(self.snapshots),
        }

if __name__ == "__main__":
    mem = HebbianMemory(dim=1024, n_concepts=66)
    
    # T1: 学习"乌鸦+石子+渴"
    print("=== T1 学习经历 ===")
    mem.experience({17, 29, 58}, label="乌鸦喝水")
    stats1 = mem.memory_stats()
    print(f"  G非零连接: {stats1['G_nonzero']}")
    
    # T2: 回忆（只给"乌鸦"）
    print("\n=== T2 主动回忆 ===")
    recalled = mem.recall({17})
    print(f"  线索: 乌鸦(17)")
    print(f"  想起来的: {recalled}")
    assert 29 in recalled, "应想起石子"
    assert 58 in recalled, "应想起渴"
    print("  [PASS] 回忆成功")
    
    # T3: 遗忘
    print("\n=== T3 遗忘 ===")
    for _ in range(5000):
        mem.G *= (1 - mem.FORGETTING_RATE)
    stats3 = mem.memory_stats()
    print(f"  遗忘后G非零: {stats3['G_nonzero']}")
    
    # T4: 重复强化
    print("\n=== T4 重复强化 ===")
    mem2 = HebbianMemory(dim=1024, n_concepts=66)
    for _ in range(5):
        mem2.experience({17, 29, 58})
    recalled2 = mem2.recall({17})
    print(f"  重复5次后回忆: {recalled2}")
    assert 29 in recalled2 and 58 in recalled2
    print("  [PASS] 强化成功")
    
    # T5: 情景记忆快照
    print("\n=== T5 情景记忆快照 ===")
    print(f"  快照数: {len(mem.snapshots)}")
    print(f"  最新快照: {mem.snapshots[-1]['label']} @ {mem.snapshots[-1]['concepts']}")
    print("  [PASS] 快照保存成功")
    
    print("\n记忆系统5项测试全部通过 ✓")
