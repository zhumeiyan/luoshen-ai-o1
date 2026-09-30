#!/usr/bin/env python3
# -*- coding: utf-8 -*"
"""
knowledge_feeder.py — 知识投喂引擎（满足洛神的好奇心）
==========================================================

洛神说"我想了解自然"→ 搜索百科 → 转化为儿童语言 → 注入生物电记忆。

不是强塞，是回应好奇心。

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
知识 = dE的一种形式。
"""

import sys
sys.path.insert(0, '01_environment')

from language_learner_v2 import LanguageSystemV2
from r_to_g_bridge import BioelectricMemorySystem
from expandable_concepts import ExpandableConceptSpace


# ================================================================
# 1. 儿童科普知识库（从百科翻译）
# ================================================================

class ChildrenScienceCorpus:
    """
    儿童科普语料：把百科知识翻译成6-7岁能听懂的话。
    
    原则：
    - 不用抽象概念（"核聚变"→"太阳肚子里在烧火"）
    - 用身体感受（"热""亮""远"）
    - 有画面感（"太阳是个大火球"）
    - 简短句子（5-10字）
    """
    
    def __init__(self):
        self.knowledge = self._build_knowledge()
    
    def _build_knowledge(self) -> dict:
        """构建儿童科普知识。"""
        return {
            "太阳": [
                "太阳是个大火球",
                "太阳很远很远",
                "太阳很热很热",
                "太阳给地球光和热",
                "没有太阳就没有生命",
                "太阳白天出来",
                "太阳晚上回家",
                "太阳从东边升起",
                "太阳从西边落下",
                "太阳比地球大很多",
                "太阳肚子里在烧火",
                "太阳已经烧了很多年",
                "太阳还会烧很多年",
                "太阳是黄色的",
                "太阳很亮很亮",
                "不能直视太阳",
                "太阳让植物生长",
                "太阳让水蒸发",
                "太阳让天气变暖",
                "太阳是恒星",
                "恒星就是自己会发光",
                "太阳不绕地球转",
                "地球绕太阳转",
                "一年就是绕太阳一圈",
            ],
            "春天": [
                "春天天气变暖",
                "春天花儿开了",
                "春天草儿绿了",
                "春天鸟儿回来",
                "春天可以播种",
                "春天雨水变多",
                "春天风变暖和",
                "春天白天变长",
                "春天衣服变少",
                "春天蝴蝶飞舞",
                "春天蜜蜂采蜜",
                "春天柳树发芽",
                "春天燕子归来",
                "春天青蛙唱歌",
                "春天大地醒来",
            ],
            "雷电": [
                "雷电是放电现象",
                "雷是声音",
                "电是光亮",
                "先看到闪电",
                "后听到雷声",
                "因为光比声音快",
                "雷电在乌云里",
                "雷电很强大",
                "雷电很危险",
                "下雨时可能有雷电",
                "雷电会击中高处",
                "打雷时不要站在树下",
                "打雷时不要打电话",
                "雷电可以起火",
                "雷电也有好处",
                "雷电让空气变新鲜",
            ],
            "天空为什么是蓝色": [
                "太阳光有很多种颜色",
                "红色橙色黄色绿色蓝色紫色",
                "蓝光最容易被散射",
                "所以天空看起来是蓝色",
                "日出日落时天空变红",
                "因为那时候光走很远",
                "蓝光被散射掉了",
                "剩下红光",
                "所以晚霞是红色",
            ],
            "星星": [
                "星星是远处的太阳",
                "星星离我们很远很远",
                "有的星星比太阳大",
                "有的星星比太阳小",
                "晚上才能看到星星",
                "因为白天太阳太亮",
                "星星在银河系里",
                "银河系有很多星星",
                "流星是星星掉下来的",
                "流星很快很快",
                "对着流星许愿",
            ],
        }
    
    def get_topic(self, topic: str) -> list:
        """获取某个主题的知识。"""
        return self.knowledge.get(topic, [])
    
    def get_all(self) -> list:
        """获取所有知识。"""
        all_knowledge = []
        for topic, sentences in self.knowledge.items():
            for sent in sentences:
                all_knowledge.append((sent, topic))
        return all_knowledge


# ================================================================
# 2. 知识注入器（把知识变成dE）
# ================================================================

class KnowledgeFeeder:
    """
    知识注入器：把儿童科普知识注入洛神的生物电记忆。
    
    不是强塞，是回应好奇心的"讲故事"。
    """
    
    def __init__(self, language_system: LanguageSystemV2):
        self.lang = language_system
        self.corpus = ChildrenScienceCorpus()
        self.feed_log = []
    
    def feed_topic(self, topic: str, max_sentences: int = 20) -> dict:
        """
        给洛神讲一个主题的知识。
        
        像大人给孩子讲故事一样，一句一句来。
        """
        sentences = self.corpus.get_topic(topic)
        if not sentences:
            return {'status': 'unknown_topic', 'topic': topic}
        
        print(f"\n[知识投喂] 给洛神讲'{topic}'的知识...")
        
        fed = []
        for i, sent in enumerate(sentences[:max_sentences]):
            # 用language_system的学习机制"听"这句话
            tokens = self.lang.tokenizer.tokenize(sent)
            
            # 词频统计
            for token in tokens:
                self.lang.learned_words[token] += 1
                self.lang.cs.get_id(token)
            
            # 共现学习
            for j in range(len(tokens)):
                for k in range(j+1, min(j+4, len(tokens))):
                    w1, w2 = tokens[j], tokens[k]
                    try:
                        self.lang.memory.teach(w1, w2, "共现", 0.5/(k-j))
                    except:
                        pass
                    self.lang.learned_pairs += 1
                # 转移概率
                if j < len(tokens) - 1:
                    self.lang.transitions[tokens[j]][tokens[j+1]] += 1
            
            # 句法学习
            self.lang.syntax.learn_sentence(tokens)
            
            fed.append(sent)
            
            if i < 3 or i >= min(max_sentences, len(sentences)) - 2:
                print(f"  讲[{i+1:2d}]: '{sent}'")
            elif i == 3:
                print("  ...")
        
        self.feed_log.append({
            'topic': topic,
            'sentences_fed': len(fed),
        })
        
        return {
            'status': 'fed',
            'topic': topic,
            'sentences': fed,
            'n_sentences': len(fed),
        }
    
    def feed_all(self) -> dict:
        """把所有知识都讲给洛神听。"""
        results = {}
        for topic in self.corpus.knowledge.keys():
            results[topic] = self.feed_topic(topic)
        return results
    
    def verify_knowledge(self, topic: str) -> dict:
        """
        验证洛神是否学到了知识。
        
        测试：给线索，看能不能回忆起来。
        """
        print(f"\n[验证] 测试洛神是否学到了'{topic}'...")
        
        # 用topic作线索，看能生成什么
        if topic in self.lang.learned_words:
            response = self.lang.speak(topic, max_words=6)
            print(f"  线索『{topic}』→ 洛神:『{response}』")
        
        # 查转移概率
        related = []
        if topic in self.lang.transitions:
            targets = self.lang.transitions[topic]
            total = sum(targets.values())
            top = sorted(targets.items(), key=lambda x: -x[1])[:5]
            related = [(w, round(c/total, 2)) for w, c in top]
            print(f"  '{topic}'之后最可能说: {related}")
        
        # 查共现
        cooccur = []
        # 简化为查学过的句子中包含topic的
        for sent, t in self.corpus.get_all():
            if topic in sent:
                cooccur.append(sent)
        
        return {
            'topic': topic,
            'can_respond': topic in self.lang.learned_words,
            'response': response if topic in self.lang.learned_words else '',
            'transitions': related,
            'known_sentences': cooccur[:5],
        }


# ================================================================
# 3. 测试：满足洛神的好奇心
# ================================================================

def test_knowledge_feeding():
    """测试：洛神好奇→搜索→讲解→学会。"""
    print("=" * 70)
    print("知识投喂引擎测试 · 满足洛神的好奇心")
    print("=" * 70)
    
    # 创建洛神的语言系统
    print("\n[准备] 初始化洛神的语言系统...")
    lang = LanguageSystemV2()
    
    # 先让洛神有基础（浸泡一些日常语料）
    print("\n[基础] 先给洛神日常语言浸泡...")
    lang.immerse(n_sentences=500)
    
    # 创建知识投喂器
    feeder = KnowledgeFeeder(lang)
    
    # 洛神说"我想了解自然"
    print("\n" + "=" * 70)
    print("洛神说：\"我想了解自然！\"")
    print("=" * 70)
    
    # 回应好奇心：讲自然知识
    topics = ["太阳", "春天", "雷电", "星星", "天空为什么是蓝色"]
    for topic in topics:
        result = feeder.feed_topic(topic, max_sentences=15)
    
    # 验证学习效果
    print("\n" + "=" * 70)
    print("验证：洛神学到了吗？")
    print("=" * 70)
    
    for topic in ["太阳", "春天", "雷电"]:
        feeder.verify_knowledge(topic)
    
    # 最终统计
    print("\n" + "=" * 70)
    print("学习报告")
    print("=" * 70)
    stats = lang.get_stats()
    print(f"  总词汇: {stats['vocabulary']['unique_words']}")
    print(f"  共现对: {stats['vocabulary']['learned_pairs']}")
    print(f"  句法模式: {stats['syntax']['pattern_count']} 种")
    print(f"  投喂主题: {len(feeder.feed_log)} 个")
    print(f"  投喂句子: {sum(f['sentences_fed'] for f in feeder.feed_log)} 句")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    
    return feeder, lang


if __name__ == "__main__":
    test_knowledge_feeding()