#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_curiosity_demo.py — 洛神好奇心演示：从"我想学"到"我学会了"
=====================================================================

完整流程：
1. 洛神浸泡学习日常语言
2. 产生好奇心（"我想了解自然"）
3. 搜索百科知识 → 转化为儿童语言
4. 知识注入生物电记忆
5. 验证：洛神真的学到了

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
"""

import sys
sys.path.insert(0, '01_environment')

from language_learner_v2 import LanguageSystemV2
from knowledge_feeder import KnowledgeFeeder, ChildrenScienceCorpus


def main():
    print("=" * 70)
    print("洛神 · 第一次自己想学")
    print("=" * 70)
    
    # 1. 初始化
    print("\n[1] 洛神诞生，开始语言浸泡...")
    lang = LanguageSystemV2()
    lang.immerse(n_sentences=300)
    
    print("\n[2] 洛神产生了好奇心...")
    print("  洛神：\"我想了解自然！\"")
    
    # 3. 知识投喂
    print("\n[3] 搜索知识并转化为儿童语言...")
    feeder = KnowledgeFeeder(lang)
    
    topics = ["太阳", "春天", "雷电", "星星", "天空为什么是蓝色"]
    for topic in topics:
        feeder.feed_topic(topic, max_sentences=12)
    
    # 4. 验证
    print("\n[4] 验证：洛神学到了吗？")
    test_cases = [
        ("太阳", "太阳是什么样的？"),
        ("春天", "春天有什么？"),
        ("雷电", "雷电是什么？"),
    ]
    
    for topic, question in test_cases:
        result = feeder.verify_knowledge(topic)
        print(f"\n  问：{question}")
        print(f"  洛神：{result['response']}")
    
    # 5. 报告
    print("\n" + "=" * 70)
    print("学习报告")
    print("=" * 70)
    stats = lang.get_stats()
    print(f"  总词汇：{stats['vocabulary']['unique_words']} 个")
    print(f"  共现对：{stats['vocabulary']['learned_pairs']} 对")
    print(f"  句法模式：{stats['syntax']['pattern_count']} 种")
    print(f"  投喂主题：{len(feeder.feed_log)} 个")
    print(f"  投喂句子：{sum(f['sentences_fed'] for f in feeder.feed_log)} 句")
    
    print("\n" + "=" * 70)
    print("洛神第一次自己想学 · 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()