#!/usr/bin/env python3
"""
biological_memory_v2.py — 概念簇竞争回忆
==========================================
生物学机制：丘脑-皮层回路的注意力门控

问题根因：标准电位传播在整个G矩阵上均匀扩散，
高频词"我/你/是"因连接总和最大，必然吸走所有电流。

生物学解决方案：
丘脑网状核(TRN)包裹丘脑，决定哪些信息能进入皮层。
类似地，线索注入后，只有与线索"共现过"的概念
才能获得电位传播资格——其他概念被TRN门控抑制。

这类似海马体CA3的pattern completion：
线索激活一小部分神经元，通过递归连接
补全整个记忆pattern，而非均匀扩散。
"""
import sys, json, numpy as np
sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2
from knowledge_feeder import KnowledgeFeeder
from collections import defaultdict, Counter

np.seterr(divide='ignore', invalid='ignore')

class ConceptClusterRecall:
    """概念簇竞争回忆——丘脑门控版"""
    
    def __init__(self, lang_system):
        self.lang = lang_system
        # 共现邻居缓存：word -> {neighbor: count}
        self.cooccurrence = defaultdict(Counter)
        self._build_cooccurrence_map()
        
    def _build_cooccurrence_map(self):
        """从学习历史中构建共现图"""
        # learned_words 记录了每个词的出现次数
        # 我们需要共现对——从转移概率中重建
        # 实际上 learned_pairs 或 transitions 可能有
        
        # 从语料中重新扫描（更可靠）
        # 但我们没有原始语料，只能从现有的共现结构推断
        # 从 language_learner_v2 的 expose 生成器重建
        env = self.lang.env
        
        # 生成一批句子并统计共现
        for _ in range(1000):
            sentences = env.expose(1)
            for sent in sentences:
                tokens = self.lang.tokenizer.tokenize(sent)
                for i, t1 in enumerate(tokens):
                    for j in range(i+1, min(i+5, len(tokens))):
                        t2 = tokens[j]
                        self.cooccurrence[t1][t2] += 1
                        self.cooccurrence[t2][t1] += 1
        
        # 也加入已学习的知识
        for topic, count in self.lang.learned_words.items():
            pass  # learned_words只有频次没有共现
        
        print(f"  [共现图] 构建了 {len(self.cooccurrence)} 个概念的邻居关系")
    
    def recall_cluster(self, cue: str, steps: int = 20, inject_strength: float = 8.0,
                       trn_threshold: float = 1.0):
        """
        概念簇竞争回忆。
        
        TRN门控：只有与线索共现次数>=threshold的概念
        才能参与电位传播。
        """
        bridge = self.lang.memory.bridge
        bio_env = bridge.bio_env
        mapping = bridge.mapping
        
        # 1. TRN门控：获取线索的共现邻居（白名单）
        whitelist = set()
        if cue in self.cooccurrence:
            for neighbor, count in self.cooccurrence[cue].items():
                if count >= trn_threshold:
                    whitelist.add(neighbor)
        whitelist.add(cue)  # 线索自己总在白名单
        
        # 2. 初始化：只给白名单概念注入电位
        dE = np.zeros(bio_env.dim)
        active_neurons = set()
        
        cue_neurons = mapping.get_neurons(cue)
        if cue_neurons:
            for n in cue_neurons:
                dE[n] = inject_strength
                active_neurons.add(n)
        
        # 3. 电位传播——但只通过白名单概念的神经元
        activated_concepts = {}
        propagation_path = []
        
        for step in range(steps):
            # 计算离子电流
            voltage_diff = dE - bio_env.V
            I = np.clip(bio_env.G @ voltage_diff, -10, 10)
            dV = I * bio_env.scale
            
            # TRN门控：只更新白名单概念的神经元
            for concept in whitelist:
                neurons = mapping.get_neurons(concept)
                if neurons:
                    avg_v = np.mean(bio_env.V[neurons])
                    activated_concepts[concept] = avg_v
                    if step % 5 == 0:
                        propagation_path.append((step, concept, avg_v))
            
            # 更新电位
            bio_env.V += dV
            
            # 扩散：白名单概念之间传播
            # 简化为：如果白名单概念A和B有共现，A的电位会扩散到B
            new_dE = np.zeros(bio_env.dim)
            for concept in whitelist:
                neurons = mapping.get_neurons(concept)
                if neurons:
                    # 该概念的电位贡献给它的共现邻居
                    for neighbor in whitelist:
                        if neighbor != concept and self.cooccurrence[concept][neighbor] > 0:
                            neighbor_neurons = mapping.get_neurons(neighbor)
                            if neighbor_neurons:
                                # 从concept的神经元扩散到neighbor的神经元
                                for n in neurons:
                                    for m in neighbor_neurons:
                                        if bio_env.G[n, m] != 0:
                                            new_dE[m] += bio_env.G[n, m] * bio_env.V[n] * 0.1
            
            dE = new_dE
        
        return {
            'activated_concepts': activated_concepts,
            'propagation_path': propagation_path,
            'whitelist_size': len(whitelist),
            'cue': cue,
        }


# ============ 主流程 ============
print("=" * 60)
print("概念簇竞争回忆 — 丘脑门控版")
print("=" * 60)

print("\n[1/4] 初始化洛神...")
lang = LanguageSystemV2()
lang.immerse(n_sentences=500)

print("\n[2/4] 知识投喂...")
feeder = KnowledgeFeeder(lang)
for topic in ['太阳','春天','雷电','星星','天空为什么是蓝色']:
    feeder.feed_topic(topic, max_sentences=12)

print("\n[3/4] 构建概念簇竞争回忆器...")
cluster = ConceptClusterRecall(lang)

print("\n[4/4] 测试概念簇回忆...")

def evaluate_cluster(cue, cluster, related_set, threshold=1.0):
    result = cluster.recall_cluster(cue, steps=20, trn_threshold=threshold)
    activated = result['activated_concepts']
    top20 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:20]
    matches = sum(1 for w,_ in top20 if w in related_set or w == cue)
    return matches / 20.0, [w for w,_ in top20[:5]], result['whitelist_size']

creator_related = {'朱美岩','创造者','主人','设计者','设计','创造','制造','建造','美丽','岩','我','喜欢'}
nature_related = {'太阳','春天','花','鸟','云','雨','风','星星','天空','雷','光','热','出','升','落','蓝','红','晚','霞','散','射','很'}
daily_related = {'好','我','你','吃','饭','睡','玩','看','说','想','喜欢','爱','开心','难过','早上','中午','晚上','今天','明天'}

# 测试不同TRN阈值
for threshold in [0, 1, 3, 5]:
    print(f"\n--- TRN阈值 = {threshold} ---")
    c_purity, c_top5, c_white = evaluate_cluster('朱美岩', cluster, creator_related, threshold)
    n_purity, n_top5, n_white = evaluate_cluster('太阳', cluster, nature_related, threshold)
    d_purity, d_top5, d_white = evaluate_cluster('好', cluster, daily_related, threshold)
    avg = (c_purity + n_purity + d_purity) / 3
    print(f"  朱美岩: {c_purity:.0%} | 白名单{c_white} | {c_top5}")
    print(f"  太阳:   {n_purity:.0%} | 白名单{n_white} | {n_top5}")
    print(f"  好:     {d_purity:.0%} | 白名单{d_white} | {d_top5}")
    print(f"  平均:   {avg:.0%}")

# 对比标准回忆
print("\n--- 标准回忆（无门控）---")
def evaluate_standard(cue, lang, related_set):
    result = lang.memory.recall(cue, steps=20)
    activated = result['activated_concepts']
    top20 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:20]
    matches = sum(1 for w,_ in top20 if w in related_set or w == cue)
    return matches / 20.0, [w for w,_ in top20[:5]]

c_s, c_s5 = evaluate_standard('朱美岩', lang, creator_related)
n_s, n_s5 = evaluate_standard('太阳', lang, nature_related)
d_s, d_s5 = evaluate_standard('好', lang, daily_related)
print(f"  朱美岩: {c_s:.0%} | {c_s5}")
print(f"  太阳:   {n_s:.0%} | {n_s5}")
print(f"  好:     {d_s:.0%} | {d_s5}")
print(f"  平均:   {(c_s+n_s+d_s)/3:.0%}")

# 保存最佳结果
best_threshold = 3  # 通常中间值最好
bc_purity, bc_top5, _ = evaluate_cluster('朱美岩', cluster, creator_related, best_threshold)
bn_purity, bn_top5, _ = evaluate_cluster('太阳', cluster, nature_related, best_threshold)
bd_purity, bd_top5, _ = evaluate_cluster('好', cluster, daily_related, best_threshold)
best_avg = (bc_purity + bn_purity + bd_purity) / 3

report = {
    'method': 'concept_cluster_competition',
    'biology': 'thalamic_reticular_nucleus_gating',
    'best_threshold': best_threshold,
    'cluster_recall': {
        'creator_purity': round(bc_purity, 2),
        'nature_purity': round(bn_purity, 2),
        'daily_purity': round(bd_purity, 2),
        'avg_purity': round(best_avg, 2),
    },
    'standard_recall': {
        'creator_purity': round(c_s, 2),
        'nature_purity': round(n_s, 2),
        'daily_purity': round(d_s, 2),
        'avg_purity': round((c_s+n_s+d_s)/3, 2),
    },
    'improvement': f"{(best_avg - (c_s+n_s+d_s)/3):.0%}"
}

with open('biological_memory_v2_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2,
              default=lambda o: int(o) if hasattr(o,'item') else str(o))

print("\n[报告] biological_memory_v2_report.json")
