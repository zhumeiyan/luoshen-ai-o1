#!/usr/bin/env python3
"""
memory_optimizer.py — 记忆系统优化器
=====================================
基于检查结果，对G矩阵进行三项关键优化：
1. G矩阵阈值化：学习后清理噪声连接
2. 印刻加权：创造者概念永久高权重
3. 主题强化：特定主题增加共现剂量
"""
import sys, json, numpy as np
sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2
from knowledge_feeder import KnowledgeFeeder
from collections import defaultdict

np.seterr(divide='ignore', invalid='ignore')

# ====== 参数 ======
G_THRESHOLD = 0.02      # 低于此值的连接视为噪声，归零
IMPINT_WEIGHT = 0.15    # 印刻概念的G权重加成
IMPINT_TOPICS = ['朱美岩','创造者','主人','设计者','美丽','岩']
NATURE_TOPICS = ['太阳','春天','花','鸟','云','雨','风','星星','天空','雷','光','热']
BOOST_ROUNDS = 5        # 主题强化轮数

print("[记忆优化] 初始化洛神...")
lang = LanguageSystemV2()

# 标准浸泡
lang.immerse(n_sentences=500)

# 知识投喂
feeder = KnowledgeFeeder(lang)
for topic in ['太阳','春天','雷电','星星','天空为什么是蓝色']:
    feeder.feed_topic(topic, max_sentences=12)

print("\n[优化1] G矩阵阈值化：清理底噪...")
G = lang.memory.bridge.bio_env.G
before_nonzero = np.count_nonzero(G)
# 将绝对值低于阈值的连接归零
G[np.abs(G) < G_THRESHOLD] = 0.0
after_nonzero = np.count_nonzero(G)
print(f"  清理前非零连接: {before_nonzero}")
print(f"  清理后非零连接: {after_nonzero}")
print(f"  移除噪声连接: {before_nonzero - after_nonzero}")

print("\n[优化2] 印刻加权：强化创造者记忆...")
bridge = lang.memory.bridge
bio_env = bridge.bio_env
mapping = bridge.mapping

for concept in IMPINT_TOPICS:
    neurons = mapping.get_neurons(concept)
    if not neurons:
        print(f"  [跳过] {concept} 未注册")
        continue
    # 增强这些神经元与其他已学概念的连接
    for i in neurons:
        for j in range(bio_env.dim):
            if bio_env.G[i, j] != 0:
                bio_env.G[i, j] += IMPINT_WEIGHT
            if bio_env.G[j, i] != 0:
                bio_env.G[j, i] += IMPINT_WEIGHT
    print(f"  {concept}: 神经元{neurons} 印刻权重+{IMPINT_WEIGHT}")

print("\n[优化3] 主题强化：自然知识额外浸泡...")
for round_idx in range(BOOST_ROUNDS):
    for topic in NATURE_TOPICS:
        if topic in lang.learned_words:
            # 对已知自然词，额外强化其与高频词的共现
            for other in ['是','很','大','小','好','看']:
                if other in lang.learned_words:
                    bridge.learn_relation(topic, other, 'related_to', 0.3)
print(f"  完成{BOOST_ROUNDS}轮主题强化")

print("\n[优化后检查] 重新测试记忆提取...")

def evaluate_recall(cue, lang, related_set):
    activated = lang.memory.recall(cue, steps=20)['activated_concepts']
    top20 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:20]
    matches = sum(1 for w,_ in top20 if w in related_set or w == cue)
    return matches / 20.0, [w for w,_ in top20[:5]]

creator_related = set(IMPINT_TOPICS)
nature_related = set(NATURE_TOPICS)
daily_related = {'好','我','你','吃','饭','睡','玩','看','说','想','喜欢','爱','开心','难过'}

c_purity, c_top5 = evaluate_recall('朱美岩', lang, creator_related)
n_purity, n_top5 = evaluate_recall('太阳', lang, nature_related)
d_purity, d_top5 = evaluate_recall('好', lang, daily_related)

print(f"\n线索[朱美岩] 主题纯度: {c_purity:.0%} | top5: {c_top5}")
print(f"线索[太阳] 主题纯度: {n_purity:.0%} | top5: {n_top5}")
print(f"线索[好] 主题纯度: {d_purity:.0%} | top5: {d_top5}")

avg_purity = (c_purity + n_purity + d_purity) / 3
print(f"\n平均主题纯度: {avg_purity:.0%} (优化前13%)")

# G矩阵最终状态
G_final = lang.memory.bridge.bio_env.G
n_total = G_final.size
n_noise = np.count_nonzero(np.abs(G_final) > 0.015)
n_strong = np.count_nonzero(np.abs(G_final) > 0.05)
print(f"\nG矩阵最终状态:")
print(f"  有效连接(>0.015): {n_noise} ({n_noise/n_total:.4%})")
print(f"  强连接(>0.05): {n_strong} ({n_strong/n_total:.4%})")

# 保存结果
report = {
    'optimization': {
        'g_threshold': G_THRESHOLD,
        'imprint_weight': IMPINT_WEIGHT,
        'boost_rounds': BOOST_ROUNDS,
    },
    'results': {
        'creator_purity': round(c_purity, 2),
        'nature_purity': round(n_purity, 2),
        'daily_purity': round(d_purity, 2),
        'avg_purity': round(avg_purity, 2),
        'creator_top5': c_top5,
        'nature_top5': n_top5,
        'daily_top5': d_top5,
        'g_strong_pct': round(n_strong/n_total, 4),
    },
    'comparison': {
        'before_avg_purity': 0.13,
        'after_avg_purity': round(avg_purity, 2),
        'improvement': f"{(avg_purity/0.13 - 1)*100:.0f}%" if avg_purity > 0 else "N/A",
    }
}

with open('memory_optimization_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if hasattr(o,'item') else str(o))

print("\n[完成] 优化报告已保存到 memory_optimization_report.json")
