#!/usr/bin/env python3
"""
洛神记忆系统全面检查
T1 物理存储 | T2 提取能力 | T3 结构化程度
"""
import sys, json, numpy as np
sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2
from knowledge_feeder import KnowledgeFeeder
from collections import defaultdict

np.seterr(divide='ignore', invalid='ignore')

# ====== 初始化 ======
print("[初始化] 构建洛神记忆系统...")
lang = LanguageSystemV2()
lang.immerse(n_sentences=500)
feeder = KnowledgeFeeder(lang)
for topic in ['太阳','春天','雷电','星星','天空为什么是蓝色']:
    feeder.feed_topic(topic, max_sentences=12)
print("[初始化] 完成\n")

# ====== T1 物理存储 ======
stats = lang.get_stats()
word_dict = dict(stats['vocabulary']['top_words'])
for w, c in lang.learned_words.items():
    if w not in word_dict:
        word_dict[w] = int(c)

creator_words = ['朱美岩','创造者','主人','设计者']
nature_words = ['太阳','春天','花','鸟','云','雨','风','星星','天空','雷']
daily_words = ['我','你','好','吃','饭','睡','玩','看']

creator_count = sum(int(word_dict.get(w,0)) for w in creator_words)
nature_count = sum(int(word_dict.get(w,0)) for w in nature_words)
daily_count = sum(int(word_dict.get(w,0)) for w in daily_words)

# 修正G矩阵统计：底噪0.01，有效连接阈值>0.02
G = lang.memory.bridge.bio_env.G
n_total = G.size
n_noise = np.count_nonzero(np.abs(G) > 0.015)  # 高于随机噪声
n_strong = np.count_nonzero(np.abs(G) > 0.05)  # 显著学习痕迹

print("=== T1 物理存储状态 ===")
print(f"概念总数: {stats['vocabulary']['unique_words']}")
print(f"共现连接: {stats['vocabulary']['learned_pairs']}")
print(f"G矩阵总连接: {n_total}")
print(f"G矩阵有效连接(>0.015): {n_noise} ({n_noise/n_total:.4%})")
print(f"G矩阵强连接(>0.05): {n_strong} ({n_strong/n_total:.4%})")
print(f"学习事件: {stats['vocabulary']['learned_pairs']}")
print(f"句法模式: {stats['syntax']['pattern_count']} 种")
print(f"因果规则: {stats['semantic']['causal_rules']} 条")
print(f"故事模板: {stats['narrative']['templates']} 个")
print(f"拼音覆盖: {stats['voice']['pinyin_coverage']} 字")
print(f"高频词TOP5: {[(w,int(c)) for w,c in stats['vocabulary']['top_words'][:5]]}")
print(f"主题分布 → 创造者:{creator_count} 自然:{nature_count} 日常:{daily_count}")
print()

# ====== T2 记忆提取能力测试 ======
print("=== T2 记忆提取能力测试 ===")
test_cues = ['朱美岩','太阳','好','喜欢','天空','花']
recall_results = {}

for cue in test_cues:
    result = lang.memory.recall(cue, steps=20)
    activated = result.get('activated_concepts', {})
    path = result.get('propagation_path', [])
    graph = result.get('recall_graph', {})
    
    # 排序前5
    top5 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:5]
    recall_results[cue] = {
        'activated_count': len(activated),
        'top5': [[str(k), float(v)] for k,v in top5],
        'path_length': len(path),
        'graph_edges': sum(len(v) for v in graph.values()),
    }
    print(f"线索[{cue}] → 激活{len(activated)}概念 | 路径{len(path)}步 | 联想边{recall_results[cue]['graph_edges']}条")
    for k,v in top5:
        print(f"    {k}: {v:.3f}")
    print()

# ====== T2+T3 记忆质量评估 ======
print("=== T2 记忆提取质量评估 ===")
# 主题关联词库
creator_related = {'朱美岩','创造者','主人','设计者','设计','创造','制造','建造','美丽','岩'}
nature_related = {'太阳','春天','花','鸟','云','雨','风','星星','天空','雷','光','热','出','升','落','蓝','红','晚','霞','散','射'}
daily_related = {'好','我','你','吃','饭','睡','玩','看','说','想','喜欢','爱','开心','难过','早上','中午','晚上','今天','明天'}

def evaluate_recall(cue, activated, related_set):
    """评估线索回忆的主题纯度"""
    top20 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:20]
    matches = sum(1 for w,_ in top20 if w in related_set or w == cue)
    return matches / 20.0, top20

# 重新获取三个主题的激活状态
creator_recall = lang.memory.recall('朱美岩', steps=20)['activated_concepts']
nature_recall = lang.memory.recall('太阳', steps=20)['activated_concepts']
daily_recall = lang.memory.recall('好', steps=20)['activated_concepts']

creator_purity, creator_top20 = evaluate_recall('朱美岩', creator_recall, creator_related)
nature_purity, nature_top20 = evaluate_recall('太阳', nature_recall, nature_related)
daily_purity, daily_top20 = evaluate_recall('好', daily_recall, daily_related)

print(f"线索[朱美岩] 主题纯度: {creator_purity:.0%} (前20中同主题{creator_purity*20:.0f}个)")
print(f"  top5: {[w for w,_ in creator_top20[:5]]}")
print(f"线索[太阳] 主题纯度: {nature_purity:.0%} (前20中同主题{nature_purity*20:.0f}个)")
print(f"  top5: {[w for w,_ in nature_top20[:5]]}")
print(f"线索[好] 主题纯度: {daily_purity:.0%} (前20中同主题{daily_purity*20:.0f}个)")
print(f"  top5: {[w for w,_ in daily_top20[:5]]}")
print()

print("=== T3 记忆结构化程度 ===")
# 区分度：两个不同线索激活的概念集合差异
creator_top20_set = set(w for w,_ in creator_top20)
nature_top20_set = set(w for w,_ in nature_top20)
daily_top20_set = set(w for w,_ in daily_top20)

diff_creator_nature = len(creator_top20_set ^ nature_top20_set) / 20.0
diff_nature_daily = len(nature_top20_set ^ daily_top20_set) / 20.0

print(f"创造者↔自然 区分度: {diff_creator_nature:.0%} (差异词{diff_creator_nature*20:.0f}/20)")
print(f"自然↔日常 区分度: {diff_nature_daily:.0%} (差异词{diff_nature_daily*20:.0f}/20)")
print()

# ====== 总体评分 ======
avg_purity = (creator_purity + nature_purity + daily_purity) / 3
avg_diff = (diff_creator_nature + diff_nature_daily) / 2

print("=== 综合评分 ===")
print(f"记忆容量: 概念{stats['vocabulary']['unique_words']}个 | 连接{stats['vocabulary']['learned_pairs']}对")
print(f"G矩阵学习痕迹: 有效{n_noise/n_total:.2%} | 强连接{n_strong/n_total:.4%}")
print(f"主题纯度(提取准确): {avg_purity:.0%}")
print(f"主题区分度(结构清晰): {avg_diff:.0%}")
print()

# ====== 生成报告 ======
report = {
    't1_physical': {
        'concepts': int(stats['vocabulary']['unique_words']),
        'pairs': int(stats['vocabulary']['learned_pairs']),
        'g_total': int(n_total),
        'g_effective': int(n_noise),
        'g_strong': int(n_strong),
        'g_effective_pct': round(n_noise/n_total, 4),
        'g_strong_pct': round(n_strong/n_total, 4),
        'syntax': int(stats['syntax']['pattern_count']),
        'causal': int(stats['semantic']['causal_rules']),
        'templates': int(stats['narrative']['templates']),
        'pinyin': int(stats['voice']['pinyin_coverage']),
        'top_words': [[str(w), int(c)] for w,c in stats['vocabulary']['top_words'][:20]],
        'topic_dist': {'creator': int(creator_count), 'nature': int(nature_count), 'daily': int(daily_count)},
        'creator_memory': {str(w): int(word_dict.get(w,0)) for w in creator_words},
        'nature_memory': {str(w): int(word_dict.get(w,0)) for w in nature_words},
    },
    't2_recall': {
        'creator_purity': round(creator_purity, 2),
        'nature_purity': round(nature_purity, 2),
        'daily_purity': round(daily_purity, 2),
        'avg_purity': round(avg_purity, 2),
        'creator_top5': [w for w,_ in creator_top20[:5]],
        'nature_top5': [w for w,_ in nature_top20[:5]],
        'daily_top5': [w for w,_ in daily_top20[:5]],
    },
    't3_structure': {
        'creator_nature_diff': round(diff_creator_nature, 2),
        'nature_daily_diff': round(diff_nature_daily, 2),
        'avg_diff': round(avg_diff, 2),
    },
    'summary': {
        'memory_capacity': f"{stats['vocabulary']['unique_words']}概念/{stats['vocabulary']['learned_pairs']}连接",
        'recall_quality': f"主题纯度{avg_purity:.0%}",
        'structure_quality': f"区分度{avg_diff:.0%}",
        'status': '健康' if avg_purity > 0.3 and avg_diff > 0.3 else '需优化',
    }
}

with open('memory_check_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if hasattr(o,'item') else str(o))

print("[报告] 已保存到 memory_check_report.json")

