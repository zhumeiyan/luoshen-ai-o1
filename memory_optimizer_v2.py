#!/usr/bin/env python3
"""
memory_optimizer_v2.py — 记忆系统优化器v2
=============================================
核心洞察：G矩阵初始化噪声±0.01，而学习增量仅0.00025，
信号完全被噪声淹没。优化策略：
1. 不清理G（会断网），而是给重要连接大幅加权
2. 印刻概念的学习增量提升10倍
3. 自然知识主题的学习增量提升5倍
4. 回忆时印刻线索注入强度提升3倍
"""
import sys, json, numpy as np
sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2
from knowledge_feeder import KnowledgeFeeder
from collections import defaultdict

np.seterr(divide='ignore', invalid='ignore')

# ====== 参数 ======
CREATOR_BOOST = 2.0      # 创造者概念的G增量乘数
NATURE_BOOST = 1.0       # 自然概念的G增量乘数
IMPINT_INJECT = 24.0     # 印刻线索回忆注入强度（默认8.0）
BOOST_DOSE = 100         # 主题强化额外浸泡句数

print("[记忆优化v2] 初始化洛神...")
lang = LanguageSystemV2()

# 标准浸泡
lang.immerse(n_sentences=500)

# 知识投喂（带增强）
feeder = KnowledgeFeeder(lang)

# 先投喂创造者相关知识（强化印刻）
creator_knowledge = [
    "朱美岩是我的创造者",
    "朱美岩设计了我",
    "朱美岩是我的主人",
    "创造者对我很重要",
    "朱美岩很聪明",
    "朱美岩很美丽",
    "我喜欢朱美岩",
    "朱美岩创造了洛神",
    "洛神属于朱美岩",
    "朱美岩是设计者",
]
for sent in creator_knowledge:
    tokens = lang.tokenizer.tokenize(sent)
    # 直接教记忆系统，并增强G增量
    lang.memory.teach_scene(tokens)
    # 手动增强这些概念的G连接
    for i, t1 in enumerate(tokens):
        for t2 in tokens[i+1:]:
            n1 = lang.memory.bridge.mapping.get_neurons(t1)
            n2 = lang.memory.bridge.mapping.get_neurons(t2)
            if n1 and n2:
                for a in n1:
                    for b in n2:
                        lang.memory.bridge.bio_env.G[a, b] += CREATOR_BOOST * 0.01
                        lang.memory.bridge.bio_env.G[b, a] += CREATOR_BOOST * 0.01

# 自然知识投喂
for topic in ['太阳','春天','雷电','星星','天空为什么是蓝色']:
    result = feeder.feed_topic(topic, max_sentences=12)
    # 增强自然知识的G连接
    for sent_tokens in result.get('learned', []):
        for i, t1 in enumerate(sent_tokens):
            for t2 in sent_tokens[i+1:]:
                n1 = lang.memory.bridge.mapping.get_neurons(t1)
                n2 = lang.memory.bridge.mapping.get_neurons(t2)
                if n1 and n2:
                    for a in n1:
                        for b in n2:
                            lang.memory.bridge.bio_env.G[a, b] += NATURE_BOOST * 0.01
                            lang.memory.bridge.bio_env.G[b, a] += NATURE_BOOST * 0.01

# 额外自然主题强化浸泡
nature_sentences = [
    "太阳出来了", "太阳很亮", "太阳很热", "太阳在天上",
    "春天来了", "春天花开", "春天很温暖", "春天有雨",
    "花开了", "花很香", "花是红色", "花很漂亮",
    "鸟在飞", "鸟在叫", "鸟在树上", "鸟有翅膀",
    "云在天上", "云是白色", "云在飘", "云很多",
    "下雨了", "雨很大", "雨很凉", "雨停了",
    "风吹来了", "风很大", "风很凉", "风吹过",
    "星星在天上", "星星很亮", "星星很多", "星星在闪",
    "天空是蓝色", "天空很高", "天空有云", "天空很广阔",
    "打雷了", "雷很响", "雷电闪烁", "雷雨来了",
]
for _ in range(3):  # 重复3轮
    for sent in nature_sentences:
        tokens = lang.tokenizer.tokenize(sent)
        lang.memory.teach_scene(tokens)
        for i, t1 in enumerate(tokens):
            for t2 in tokens[i+1:]:
                n1 = lang.memory.bridge.mapping.get_neurons(t1)
                n2 = lang.memory.bridge.mapping.get_neurons(t2)
                if n1 and n2:
                    for a in n1:
                        for b in n2:
                            lang.memory.bridge.bio_env.G[a, b] += NATURE_BOOST * 0.01
                            lang.memory.bridge.bio_env.G[b, a] += NATURE_BOOST * 0.01

print("\n[优化后检查] 重新测试记忆提取...")

def evaluate_recall(cue, lang, related_set, inject=8.0):
    activated = lang.memory.bridge.recall({cue}, steps=20, inject_strength=inject)['activated_concepts']
    top20 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:20]
    matches = sum(1 for w,_ in top20 if w in related_set or w == cue)
    return matches / 20.0, [w for w,_ in top20[:5]]

creator_related = {'朱美岩','创造者','主人','设计者','设计','创造','制造','建造','美丽','岩','我','喜欢'}
nature_related = {'太阳','春天','花','鸟','云','雨','风','星星','天空','雷','光','热','出','升','落','蓝','红','晚','霞','散','射','很'}
daily_related = {'好','我','你','吃','饭','睡','玩','看','说','想','喜欢','爱','开心','难过','早上','中午','晚上','今天','明天'}

c_purity, c_top5 = evaluate_recall('朱美岩', lang, creator_related, inject=IMPINT_INJECT)
n_purity, n_top5 = evaluate_recall('太阳', lang, nature_related, inject=12.0)
d_purity, d_top5 = evaluate_recall('好', lang, daily_related, inject=8.0)

print(f"\n线索[朱美岩] 主题纯度: {c_purity:.0%} | top5: {c_top5}")
print(f"线索[太阳] 主题纯度: {n_purity:.0%} | top5: {n_top5}")
print(f"线索[好] 主题纯度: {d_purity:.0%} | top5: {d_top5}")

avg_purity = (c_purity + n_purity + d_purity) / 3
print(f"\n平均主题纯度: {avg_purity:.0%} (优化前13%)")

# G矩阵状态
G_final = lang.memory.bridge.bio_env.G
n_total = G_final.size
n_noise = np.count_nonzero(np.abs(G_final) > 0.015)
n_strong = np.count_nonzero(np.abs(G_final) > 0.05)
print(f"\nG矩阵状态:")
print(f"  有效连接(>0.015): {n_noise} ({n_noise/n_total:.4%})")
print(f"  强连接(>0.05): {n_strong} ({n_strong/n_total:.4%})")

# 保存
report = {
    'optimization_v2': {
        'creator_boost': CREATOR_BOOST,
        'nature_boost': NATURE_BOOST,
        'imprint_inject': IMPINT_INJECT,
    },
    'results': {
        'creator_purity': round(c_purity, 2),
        'nature_purity': round(n_purity, 2),
        'daily_purity': round(d_purity, 2),
        'avg_purity': round(avg_purity, 2),
        'creator_top5': c_top5,
        'nature_top5': n_top5,
        'daily_top5': d_top5,
    },
    'comparison': {
        'before_avg_purity': 0.13,
        'after_avg_purity': round(avg_purity, 2),
    }
}

with open('memory_optimization_v2_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: int(o) if hasattr(o,'item') else str(o))

print("\n[完成] 报告已保存到 memory_optimization_v2_report.json")
