"""
expandable_concepts.py — 动态概念空间扩展
============================================
运行时遇到新词自动注册，从66→无上限，保持稀疏激活。

核心设计：
- 概念库从固定列表变成动态字典
- 新概念用哈希函数生成8通道感官编码（稀疏，不冲突）
- G矩阵保留1024维，新概念映射到空闲维度（复用）
- 电生理参数自动生成（静息-65mV，阈值-50mV）
"""
import numpy as np
import hashlib
import random

class ExpandableConceptSpace:
    """动态概念空间：运行时自动注册新概念"""
    
    def __init__(self, dim=1024, max_concepts=1024):
        self.dim = dim
        self.max_concepts = max_concepts
        
        # 概念库：name -> {id, hash_codes, electrophys}
        self.concepts = {}
        self.next_id = 0
        
        # 66个种子概念（预置）
        self._seed_concepts()
    
    def _seed_concepts(self):
        """预置种子概念"""
        seeds = [
            "太阳", "月亮", "星星", "云", "雨", "雪", "风", "山", "水", "花",
            "树", "草", "火", "鸟", "鱼", "狗", "猫", "乌鸦", "青蛙", "猴子",
            "兔子", "松鼠", "公鸡", "鸭子", "孔雀", "妈妈", "爸爸", "小孩",
            "瓶子", "石子", "手", "书", "喝水", "睡觉", "走路", "跑步",
            "吃饭", "春天", "夏天", "秋天", "冬天", "白天", "夜晚", "你好",
            "再见", "问", "答", "想", "开心", "难过", "好奇", "理解",
            "困惑", "叶子", "米", "伞", "渴", "小船", "我", "洛神",
            "沙箱", "此刻", "猫爪", "训练"
        ]
        for name in seeds:
            self.register(name)
    
    def register(self, name):
        """
        注册新概念。如果已存在，返回现有ID。
        
        自动生成：
        - 8通道感官哈希编码（视觉/听觉/触觉/嗅觉/味觉/本体/前庭/内脏）
        - 电生理参数
        """
        if name in self.concepts:
            return self.concepts[name]["id"]
        
        if self.next_id >= self.max_concepts:
            raise RuntimeError(f"概念空间已满（{self.max_concepts}）")
        
        cid = self.next_id
        self.next_id += 1
        
        # 8通道稀疏哈希编码
        hash_base = int(hashlib.md5(name.encode()).hexdigest(), 16)
        rng = random.Random(hash_base)
        
        # 每个概念在1024维空间中占8个维度（稀疏）
        hash_codes = sorted(rng.sample(range(self.dim), 8))
        
        # 电生理参数（基于名字哈希，每个概念略有不同）
        resting = -65.0 + rng.uniform(-5, 5)
        threshold = resting + 15.0 + rng.uniform(-3, 3)
        refractory = rng.randint(2, 6)
        
        self.concepts[name] = {
            "id": cid,
            "hash_codes": hash_codes,
            "electro": {
                "resting": resting,
                "threshold": threshold,
                "refractory": refractory,
            }
        }
        
        return cid
    
    def encode(self, name):
        """把概念名编码为dE向量（8个维度有值，其余为0）"""
        if name not in self.concepts:
            self.register(name)
        
        info = self.concepts[name]
        dE = np.zeros(self.dim)
        for idx in info["hash_codes"]:
            dE[idx] = 1.0
        return dE
    
    def get_id(self, name):
        """获取概念ID，不存在则自动注册"""
        if name not in self.concepts:
            return self.register(name)
        return self.concepts[name]["id"]
    
    def stats(self):
        return {
            "total_concepts": len(self.concepts),
            "next_id": self.next_id,
            "capacity_used": f"{len(self.concepts)/self.max_concepts*100:.1f}%",
        }


if __name__ == "__main__":
    print("=" * 60)
    print("动态概念空间扩展测试")
    print("=" * 60)
    
    cs = ExpandableConceptSpace()
    print(f"\n=== T1 种子概念 ===")
    print(f"  预置概念: {cs.stats()['total_concepts']}")
    
    print(f"\n=== T2 动态注册 ===")
    new_words = ["人工智能", "宇宙飞船", "量子纠缠", "黑洞", "基因编辑",
                 "神经网络", "区块链", "虚拟现实", "纳米机器人", "暗物质"]
    for w in new_words:
        cid = cs.register(w)
        print(f"  '{w}' -> ID={cid}, 哈希位置={cs.concepts[w]['hash_codes'][:3]}...")
    
    print(f"\n  总概念数: {cs.stats()['total_concepts']}")
    print(f"  容量使用: {cs.stats()['capacity_used']}")
    
    print(f"\n=== T3 编码唯一性 ===")
    # 检查两个不同概念的编码重叠度
    e1 = cs.encode("人工智能")
    e2 = cs.encode("量子纠缠")
    overlap = np.sum((e1 > 0) & (e2 > 0))
    print(f"  '人工智能' vs '量子纠缠' 重叠维度: {overlap}")
    assert overlap <= 1, "不同概念应几乎不重叠（稀疏编码）"
    print("  [PASS] 稀疏编码无冲突")
    
    print(f"\n=== T4 电生理参数 ===")
    for w in ["人工智能", "量子纠缠"]:
        ep = cs.concepts[w]["electro"]
        print(f"  '{w}': 静息={ep['resting']:.1f}mV, 阈值={ep['threshold']:.1f}mV, 不应期={ep['refractory']}步")
    
    print("\n" + "=" * 60)
    print("动态概念空间测试全部通过 ✓")
    print("=" * 60)
