#!/usr/bin/env python3
"""
biological_memory_enhancer.py — 生物学记忆增强器
=====================================================
用三项生物学机制解决主题纯度问题：

1. 侧向抑制(Lateral Inhibition)：
   类似皮层抑制性中间神经元(IN)。
   当高频词("我/是/很")被激活时，释放抑制信号
   到相邻神经元集群，给主题概念留出空间。

2. 稀疏编码(Sparse Coding)：
   类似海马体CA3的稀疏激活(约2-4%)。
   每次回忆只让最强的前K%概念保留电位，
   其余归零——防止电位均匀扩散。

3. 乙酰胆碱调制(ACh Modulation)：
   线索注入时临时增强该线索的突触权重，
   类似注意状态下的神经调制。

4. 印刻蛋白质标签(Synaptic Tagging)：
   创造者概念获得"蛋白质合成"标签，
   其突触增强更难衰减、更难被覆盖。
"""
import sys, json, numpy as np
sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2
from knowledge_feeder import KnowledgeFeeder
from collections import defaultdict

np.seterr(divide='ignore', invalid='ignore')

# ============ 生物学参数 ============
LATERAL_INHIBITION = 0.3    # 侧向抑制强度
SPARSE_TOP_PCT = 0.15       # 稀疏编码：只保留前15%激活概念
ACh_BOOST = 3.0             # 乙酰胆碱增强：线索突触权重临时×3
PROTEIN_TAG_DECAY = 0.95    # 蛋白质标签衰减（每轮）

class BiologicalMemoryEnhancer:
    """生物学记忆增强器——包裹在现有记忆系统之上"""
    
    def __init__(self, lang_system):
        self.lang = lang_system
        # 蛋白质标签：标记哪些突触被"印刻"
        # protein_tags[(pre, post)] = tag_strength
        self.protein_tags = defaultdict(float)
        # 印刻概念列表
        self.imprinted_concepts = set()
        
    def imprint(self, concept: str, tag_strength: float = 1.0):
        """
        印刻：给概念打上蛋白质合成标签。
        类似早期经验对突触的永久标记。
        """
        self.imprinted_concepts.add(concept)
        neurons = self.lang.memory.bridge.mapping.get_neurons(concept)
        if not neurons:
            return
        # 标记这些神经元所有传出连接的蛋白质标签
        for n in neurons:
            for j in range(self.lang.memory.bridge.bio_env.dim):
                self.protein_tags[(n, j)] = max(
                    self.protein_tags[(n, j)], tag_strength
                )
        print(f"  [印刻] {concept} 蛋白质标签强度={tag_strength}")
    
    def _apply_lateral_inhibition(self, activated: dict, inhibition: float = 0.3):
        """
        侧向抑制：强激活概念抑制弱激活概念。
        
        生物学基础：皮层锥体细胞激活时，
         basket细胞/ chandelier细胞释放GABA，
         抑制相邻锥体细胞——产生" winner-take-all"效果。
        """
        if not activated:
            return activated
        
        # 按激活度排序
        sorted_items = sorted(activated.items(), key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(sorted_items) * SPARSE_TOP_PCT))
        
        # 只保留前15%的强者，其余被抑制归零
        winners = dict(sorted_items[:n_keep])
        
        # 赢者对输者的侧向抑制
        # 每个赢者的激活度按inhibition比例削弱其他概念
        # 但这里我们已经把输者归零了，所以抑制体现在：
        # 赢者之间也互相抑制（避免同时激活太多）
        for i, (w1, v1) in enumerate(sorted_items[:n_keep]):
            for w2, v2 in sorted_items[:n_keep]:
                if w1 != w2:
                    # 互相抑制
                    winners[w1] -= inhibition * v2 / n_keep
        
        # 确保不为负
        for w in winners:
            winners[w] = max(0.0, winners[w])
        
        return winners
    
    def recall_bio(self, cue: str, steps: int = 20, inject_strength: float = 8.0):
        """
        生物学增强回忆：ACh调制 + 侧向抑制 + 稀疏编码。
        """
        bridge = self.lang.memory.bridge
        bio_env = bridge.bio_env
        mapping = bridge.mapping
        
        # 1. ACh调制：线索概念的突触权重临时增强
        cue_neurons = mapping.get_neurons(cue)
        ach_backup = {}
        if cue_neurons:
            for n in cue_neurons:
                for j in range(bio_env.dim):
                    key = (n, j)
                    if key in self.protein_tags:
                        # 印刻突触获得额外ACh增强
                        ach_backup[key] = bio_env.G[n, j]
                        bio_env.G[n, j] *= (ACh_BOOST + self.protein_tags[key])
                    else:
                        ach_backup[key] = bio_env.G[n, j]
                        bio_env.G[n, j] *= ACh_BOOST
        
        # 2. 标准电位传播回忆
        result = bridge.recall({cue}, steps=steps, inject_strength=inject_strength)
        activated = result['activated_concepts']
        
        # 3. 侧向抑制 + 稀疏编码
        activated_sparse = self._apply_lateral_inhibition(
            activated, inhibition=LATERAL_INHIBITION
        )
        
        # 4. 恢复ACh调制（临时效应消退）
        for key, orig_val in ach_backup.items():
            bio_env.G[key[0], key[1]] = orig_val
        
        result['activated_concepts'] = activated_sparse
        result['n_sparse'] = len(activated_sparse)
        return result


# ============ 主流程 ============
print("=" * 60)
print("生物学记忆增强器")
print("=" * 60)

# 1. 初始化洛神
print("\n[1/5] 初始化洛神...")
lang = LanguageSystemV2()
lang.immerse(n_sentences=500)

# 2. 知识投喂
print("\n[2/5] 知识投喂...")
feeder = KnowledgeFeeder(lang)
for topic in ['太阳','春天','雷电','星星','天空为什么是蓝色']:
    feeder.feed_topic(topic, max_sentences=12)

# 3. 创建增强器并印刻
print("\n[3/5] 生物学印刻...")
enhancer = BiologicalMemoryEnhancer(lang)

# 创造者印刻（蛋白质标签）
creator_concepts = ['朱美岩','创造者','主人','设计者']
for concept in creator_concepts:
    enhancer.imprint(concept, tag_strength=2.0)

# 自然知识印刻
nature_concepts = ['太阳','春天','花','鸟','云','雨','风','星星','天空','雷']
for concept in nature_concepts:
    enhancer.imprint(concept, tag_strength=1.0)

print(f"\n  印刻概念总数: {len(enhancer.imprinted_concepts)}")
print(f"  蛋白质标签总数: {len(enhancer.protein_tags)}")

# 4. 测试提取
print("\n[4/5] 测试生物学增强回忆...")

def evaluate_bio_recall(cue, enhancer, related_set):
    result = enhancer.recall_bio(cue, steps=20, inject_strength=12.0)
    activated = result['activated_concepts']
    top20 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:20]
    matches = sum(1 for w,_ in top20 if w in related_set or w == cue)
    return matches / 20.0, [w for w,_ in top20[:5]], len(activated)

creator_related = {'朱美岩','创造者','主人','设计者','设计','创造','制造','建造','美丽','岩','我','喜欢'}
nature_related = {'太阳','春天','花','鸟','云','雨','风','星星','天空','雷','光','热','出','升','落','蓝','红','晚','霞','散','射','很'}
daily_related = {'好','我','你','吃','饭','睡','玩','看','说','想','喜欢','爱','开心','难过','早上','中午','晚上','今天','明天'}

c_purity, c_top5, c_n = evaluate_bio_recall('朱美岩', enhancer, creator_related)
n_purity, n_top5, n_n = evaluate_bio_recall('太阳', enhancer, nature_related)
d_purity, d_top5, d_n = evaluate_bio_recall('好', enhancer, daily_related)

print(f"\n  线索[朱美岩] 纯度:{c_purity:.0%} | 稀疏激活{c_n}概念 | top5: {c_top5}")
print(f"  线索[太阳] 纯度:{n_purity:.0%} | 稀疏激活{n_n}概念 | top5: {n_top5}")
print(f"  线索[好]   纯度:{d_purity:.0%} | 稀疏激活{d_n}概念 | top5: {d_top5}")

avg_purity = (c_purity + n_purity + d_purity) / 3
print(f"\n  平均主题纯度: {avg_purity:.0%}")

# 5. 对比标准回忆
print("\n[5/5] 对比标准回忆（无生物学增强）...")

def evaluate_standard(cue, lang, related_set):
    result = lang.memory.recall(cue, steps=20)
    activated = result['activated_concepts']
    top20 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:20]
    matches = sum(1 for w,_ in top20 if w in related_set or w == cue)
    return matches / 20.0, [w for w,_ in top20[:5]]

c_std, c_std5 = evaluate_standard('朱美岩', lang, creator_related)
n_std, n_std5 = evaluate_standard('太阳', lang, nature_related)
d_std, d_std5 = evaluate_standard('好', lang, daily_related)

print(f"\n  标准[朱美岩] 纯度:{c_std:.0%} | top5: {c_std5}")
print(f"  标准[太阳]   纯度:{n_std:.0%} | top5: {n_std5}")
print(f"  标准[好]     纯度:{d_std:.0%} | top5: {d_std5}")

avg_std = (c_std + n_std + d_std) / 3
print(f"\n  标准平均纯度: {avg_std:.0%}")

# 报告
report = {
    'mechanisms': {
        'lateral_inhibition': LATERAL_INHIBITION,
        'sparse_top_pct': SPARSE_TOP_PCT,
        'ach_boost': ACh_BOOST,
        'protein_tag_decay': PROTEIN_TAG_DECAY,
    },
    'imprint': {
        'n_imprinted': len(enhancer.imprinted_concepts),
        'n_protein_tags': len(enhancer.protein_tags),
        'creator_concepts': creator_concepts,
        'nature_concepts': nature_concepts,
    },
    'biological_recall': {
        'creator_purity': round(c_purity, 2),
        'nature_purity': round(n_purity, 2),
        'daily_purity': round(d_purity, 2),
        'avg_purity': round(avg_purity, 2),
        'creator_top5': c_top5,
        'nature_top5': n_top5,
        'daily_top5': d_top5,
    },
    'standard_recall': {
        'creator_purity': round(c_std, 2),
        'nature_purity': round(n_std, 2),
        'daily_purity': round(d_std, 2),
        'avg_purity': round(avg_std, 2),
        'creator_top5': c_std5,
        'nature_top5': n_std5,
        'daily_top5': d_std5,
    },
    'improvement': {
        'creator_delta': f"{(c_purity - c_std):.0%}",
        'nature_delta': f"{(n_purity - n_std):.0%}",
        'daily_delta': f"{(d_purity - d_std):.0%}",
        'avg_delta': f"{(avg_purity - avg_std):.0%}",
    }
}

with open('biological_memory_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, 
              default=lambda o: int(o) if hasattr(o,'item') else str(o))

print("\n" + "=" * 60)
print(f"生物学增强: 平均纯度 {avg_purity:.0%}")
print(f"标准回忆:   平均纯度 {avg_std:.0%}")
print(f"提升:       {(avg_purity - avg_std):.0%}")
print("=" * 60)
print("\n[报告] biological_memory_report.json")
