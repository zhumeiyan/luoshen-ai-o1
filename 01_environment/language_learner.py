#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
language_learner.py — 自主语言学习模块
========================================

洛神学说话：不是被灌输知识，是在语言环境里"泡"出来的。

生物学原型：
- 婴儿学习语言：不是上课，是每天8小时暴露在母语中
- 底层机制：听到→感知→统计规律→预测→说→验证→修正
- 关键：自主归纳，不是被动接收

设计：
1. LanguageEnvironment：语料流环境（模拟成人对婴儿说话）
2. LanguageLearner：听→记→归纳（赫布共现→G电导增强）
3. LanguageSpeaker：说→给线索→电位传播→输出句子

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
语言学习：把dE变为"听到的话"，让洛神自己发现规律。

核心原理：
- 共现学习：两个词在相邻时间窗口都激活 → G连接增强
- 预测能力：听到"今天" → G传播 → "天气"电位升高
- 序列记忆：词A→词B→词C的时序 → 形成链式G连接
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expandable_concepts import ExpandableConceptSpace
from r_to_g_bridge import BioelectricMemorySystem


# ================================================================
# 1. 语言环境（语料流）
# ================================================================

class LanguageEnvironment:
    """
    语言环境：模拟人类对婴儿说话。
    
    语料按场景分类（日常/情感/需求/描述），
    洛神在语言浸泡中自主发现规律。
    """
    
    def __init__(self):
        self.corpus = self._build_corpus()
        self.exposure_count = 0
        self.history = deque(maxlen=1000)  # 暴露历史
        
    def _build_corpus(self) -> Dict[str, List[str]]:
        """构建语料库——模拟婴儿日常听到的中文环境（400+句）。"""
        return {
            "日常问候": [
                "你好", "早上好", "晚上好", "吃了吗", "今天天气不错", "最近怎么样",
                "好久不见", "再见", "晚安", "你好啊", "早上好呀", "晚安好梦",
                "中午好", "下午好", "你好吗", "我很好", "谢谢", "不客气",
                "很高兴见到你", "下次见", "拜拜", "回头见", "你回来啦",
                "早安", "午安", "问候", "打招呼", "微笑", "点头",
            ],
            "表达需求": [
                "我饿了", "我想喝水", "我困了", "我想出去玩", "帮帮我",
                "我想睡觉", "我想吃饭", "我渴了", "我想要", "给我",
                "我要这个", "我不要", "我累了", "我想休息", "我要妈妈",
                "我要爸爸", "我要抱抱", "我要玩具", "我要牛奶", "我要水果",
                "我冷了", "我热了", "我不舒服", "我想尿尿", "我想便便",
                "我要看书", "我要画画", "我要唱歌", "我要跳舞", "我要洗澡",
                "我要刷牙", "我要穿衣服", "我要鞋子", "我要帽子", "我要出去",
            ],
            "情感表达": [
                "我开心", "我难过", "我害怕", "我喜欢你", "我爱你",
                "我好累", "真高兴", "太好了", "太棒了", "伤心",
                "害怕", "开心", "喜欢", "爱", "生气", "惊讶",
                "兴奋", "无聊", "好奇", "困惑", "满足", "失望",
                "骄傲", "害羞", "嫉妒", "同情", "感激", "感动",
                "幸福", "痛苦", "舒服", "紧张", "放松", "期待",
                "想念", "讨厌", "烦", "棒", "好", "坏",
            ],
            "描述事物": [
                "这是苹果", "那是桌子", "天是蓝的", "花是红的",
                "水很凉", "太阳很热", "月亮在天上", "星星很亮",
                "小鸟在飞", "鱼在水里", "猫在跑", "狗在叫",
                "树很高", "草很绿", "云很白", "雨在下",
                "风在吹", "雪很冷", "火很烫", "冰很硬",
                "书很重", "球很圆", "车很快", "船在飘",
                "房子很大", "门开了", "窗关了", "灯亮了",
                "电视在响", "音乐好听", "颜色漂亮", "味道香",
                "声音大", "重量轻", "形状怪", "大小合适",
            ],
            "简单问答": [
                "这是什么", "那是什么", "为什么", "怎么办",
                "在哪里", "是谁", "怎么做", "可以吗",
                "好不好", "对不对", "什么时候", "多少钱",
                "有多少", "有多远", "有多大", "有多重",
                "真的吗", "假的吗", "是吗", "不是吗",
                "知道吗", "明白吗", "懂吗", "会吗",
                "能吗", "要吗", "想吗", "喜欢吗",
                "害怕吗", "开心吗", "难过吗", "饿了吗",
            ],
            "创造者专属": [
                "你好朱美岩", "朱美岩早上好", "创造者你好", "我的主人",
                "设计者在吗", "朱美岩晚安", "我的创造者", "朱美岩我饿",
                "朱美岩我想玩", "朱美岩谢谢你", "朱美岩我爱你",
                "朱美岩在吗", "朱美岩我困了", "朱美岩我开心",
                "朱美岩我害怕", "朱美岩帮我", "朱美岩晚安好梦",
                "创造者早安", "创造者午安", "创造者下午好",
                "主人你好", "主人早上好", "主人晚安",
                "设计者谢谢你", "设计者我爱你", "设计者我饿了",
                "我的创造者你好", "我的主人早上好",
                "朱美岩今天天气不错", "朱美岩吃了吗",
                "创造者最近怎么样", "主人好久不见",
            ],
            "动作指令": [
                "过来", "坐下", "站起来", "走", "跑", "跳",
                "拿", "放", "打开", "关上", "推", "拉",
                "看", "听", "说", "笑", "哭", "叫",
                "吃", "喝", "睡", "醒", "玩", "学",
                "画", "写", "读", "唱", "跳", "跑",
                "停", "开始", "结束", "继续", "重复",
                "快", "慢", "高", "低", "大", "小",
                "上", "下", "左", "右", "前", "后",
            ],
            "社交互动": [
                "一起玩", "分享", "轮流", "等待", "排队",
                "谢谢帮助", "对不起", "没关系", "请",
                "你好朋友", "再见朋友", "朋友你好",
                "我们一起", "大家一起来", "合作",
                "握手", "拥抱", "拍手", "挥手",
                "微笑打招呼", "点头问好", "鞠躬",
                "送礼物", "收礼物", "分享食物",
                "邀请", "接受", "拒绝", "同意",
                "不同意", "商量", "讨论", "聊天",
            ],
            "认知学习": [
                "这是红色", "这是蓝色", "这是黄色", "这是绿色",
                "一", "二", "三", "四", "五",
                "大", "小", "多", "少", "长", "短",
                "高", "矮", "胖", "瘦", "快", "慢",
                "上", "下", "左", "右", "里", "外",
                "前", "后", "早", "晚", "今天", "明天",
                "昨天", "现在", "以后", "刚才",
                "早上", "中午", "下午", "晚上",
                "春天", "夏天", "秋天", "冬天",
                "星期一", "星期二", "星期三",
            ],
            "自然观察": [
                "太阳出来了", "月亮出来了", "星星在眨眼",
                "云在飘", "雨在下", "风在吹", "雪在飘",
                "花开", "花落", "草长", "树摇",
                "鸟叫", "虫鸣", "鱼游", "猫跳",
                "狗跑", "马走", "牛吃草", "羊叫",
                "天亮了", "天黑了", "日出", "日落",
                "彩虹", "闪电", "雷声", "雾",
                "露水", "霜", "冰", "火",
            ],
        }
    
    def expose(self, n_sentences: int = 1) -> List[str]:
        """
        暴露n句话给洛神。
        每次从随机场景随机抽取，模拟真实语言环境的不规则性。
        """
        sentences = []
        for _ in range(n_sentences):
            scene = random.choice(list(self.corpus.keys()))
            sentence = random.choice(self.corpus[scene])
            sentences.append(sentence)
            self.history.append({
                'sentence': sentence,
                'scene': scene,
                'step': self.exposure_count
            })
            self.exposure_count += 1
        return sentences
    
    def get_exposure_stats(self) -> Dict:
        """暴露统计。"""
        scene_counts = defaultdict(int)
        for h in self.history:
            scene_counts[h['scene']] += 1
        return {
            'total_exposures': self.exposure_count,
            'unique_sentences': len(set(h['sentence'] for h in self.history)),
            'scene_distribution': dict(scene_counts),
        }


# ================================================================
# 2. 语言学习者（听→记→归纳）
# ================================================================

class LanguageLearner:
    """
    洛神的语言学习引擎。
    
    不是被动接收，是主动感知：
    - 听到一句话 → dE注入 → P/F感知模式
    - 相邻词共现 → G电导增强（赫布学习）
    - 多次暴露后 → 形成词与词的预测链
    """
    
    def __init__(self, memory: BioelectricMemorySystem,
                 concept_space: ExpandableConceptSpace,
                 window_size: int = 3):
        self.memory = memory
        self.cs = concept_space
        self.window_size = window_size  # 共现窗口
        
        # 语言统计（自主归纳）
        self.word_freq = defaultdict(int)        # 词频
        self.cooccur = defaultdict(lambda: defaultdict(float))  # 共现矩阵
        self.transition = defaultdict(lambda: defaultdict(float))  # 转移概率
        self.context_map = defaultdict(list)     # 语境映射
        
        # 学习记录
        self.learned_pairs = 0
        self.exposure_log = deque(maxlen=500)
        
    def listen(self, sentence: str) -> Dict:
        """
        听一句话，自主学习。
        
        步骤：
        1. 分词 → 概念编码
        2. 词频统计
        3. 窗口内共现 → G连接增强
        4. 序列转移记录
        """
        # 分词（按2字切分中文）
        tokens = self._tokenize(sentence)
        if len(tokens) < 1:
            return {'status': 'empty', 'tokens': []}
        
        # 1. 确保所有词都在概念库
        for token in tokens:
            self.cs.get_id(token)
        
        # 2. 词频统计
        for token in tokens:
            self.word_freq[token] += 1
        
        # 3. 窗口内共现学习（赫布：同时激活→连接增强）
        new_pairs = 0
        for i in range(len(tokens)):
            for j in range(i+1, min(i+self.window_size+1, len(tokens))):
                w1, w2 = tokens[i], tokens[j]
                # 共现计数
                self.cooccur[w1][w2] += 1.0 / (j - i)  # 距离越近权重越高
                self.cooccur[w2][w1] += 1.0 / (j - i)
                
                # R→G桥接：教给生物电记忆（若容量满则降级为纯统计）
                try:
                    self.memory.teach(w1, w2, "共现", 0.5 / (j - i))
                    new_pairs += 1
                except RuntimeError:
                    # 概念容量已满，仅做统计不写入物理G矩阵
                    pass
        
        # 4. 序列转移（A→B的顺序）
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            self.transition[w1][w2] += 1.0
        
        # 5. 整句作为语境
        if len(tokens) >= 2:
            first = tokens[0]
            rest = "".join(tokens[1:])
            self.context_map[first].append(rest)
        
        self.learned_pairs += new_pairs
        self.exposure_log.append({
            'sentence': sentence,
            'tokens': tokens,
            'new_pairs': new_pairs
        })
        
        return {
            'status': 'learned',
            'tokens': tokens,
            'new_pairs': new_pairs,
            'total_pairs': self.learned_pairs,
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        智能中文分词：
        - 保留完整名字（朱美岩→不拆分）
        - 其他连续汉字按2字切分
        - 单字保留
        """
        tokens = []
        i = 0
        while i < len(text):
            if '\u4e00' <= text[i] <= '\u9fff':
                # 连续汉字
                j = i
                while j < len(text) and '\u4e00' <= text[j] <= '\u9fff':
                    j += 1
                word = text[i:j]
                
                # 检查是否包含名字（保留完整）
                if '朱美岩' in word:
                    # 拆分名字前后的字
                    parts = word.split('朱美岩')
                    for idx, part in enumerate(parts):
                        # 处理名字前的部分
                        for k in range(0, len(part), 2):
                            chunk = part[k:k+2]
                            if chunk:
                                tokens.append(chunk)
                        # 插入名字
                        if idx < len(parts) - 1:
                            tokens.append('朱美岩')
                else:
                    # 普通汉字，按2字切分
                    for k in range(0, len(word), 2):
                        chunk = word[k:k+2]
                        if chunk:
                            tokens.append(chunk)
                i = j
            else:
                i += 1
        return tokens
    
    def get_language_stats(self) -> Dict:
        """获取自主学习成果统计。"""
        # 计算转移概率
        transition_probs = {}
        for w1, targets in self.transition.items():
            total = sum(targets.values())
            if total > 0:
                transition_probs[w1] = {
                    w2: round(count/total, 3) 
                    for w2, count in sorted(targets.items(), key=lambda x: -x[1])[:5]
                }
        
        # 高频词
        top_words = sorted(self.word_freq.items(), key=lambda x: -x[1])[:15]
        
        # 最强共现对
        top_cooccur = []
        for w1, targets in self.cooccur.items():
            for w2, strength in targets.items():
                if w1 < w2:  # 避免重复
                    top_cooccur.append((w1, w2, strength))
        top_cooccur = sorted(top_cooccur, key=lambda x: -x[2])[:10]
        
        return {
            'total_exposures': len(self.exposure_log),
            'unique_words': len(self.word_freq),
            'learned_pairs': self.learned_pairs,
            'top_words': top_words,
            'transition_probs': transition_probs,
            'top_cooccur': top_cooccur,
        }
    
    def can_understand(self, word: str) -> bool:
        """检查是否'认识'一个词（有共现记录）。"""
        return word in self.word_freq and self.word_freq[word] >= 2


# ================================================================
# 3. 语言说话者（说→给线索→电位传播→输出）
# ================================================================

class LanguageSpeaker:
    """
    洛神"说话"的能力。
    
    不是查字典输出，是：
    1. 接收意图（一个词作为dE线索）
    2. 电位在G网络中传播
    3. 激活的序列 = "说出来的话"
    """
    
    def __init__(self, learner: LanguageLearner):
        self.learner = learner
        
    def speak(self, intent: str, max_words: int = 5) -> Dict:
        """
        给定意图，'说'一句话。
        
        机制：
        1. intent作为起点
        2. 查转移概率找最可能的下一个词
        3. 依次生成序列
        """
        if intent not in self.learner.word_freq:
            return {'sentence': intent, 'generated': [intent], 'confidence': 0.0}
        
        sequence = [intent]
        current = intent
        total_confidence = 0.0
        
        for _ in range(max_words - 1):
            # 找下一个最可能的词
            if current not in self.learner.transition:
                break
            
            targets = self.learner.transition[current]
            if not targets:
                break
            
            # 按概率选择
            total = sum(targets.values())
            candidates = [(w, c/total) for w, c in targets.items()]
            candidates.sort(key=lambda x: -x[1])
            
            # 选top3中随机（避免总是重复同一句）
            top3 = candidates[:3]
            words, probs = zip(*top3)
            next_word = random.choices(words, weights=probs)[0]
            
            sequence.append(next_word)
            total_confidence += probs[0]
            current = next_word
            
            # 避免循环
            if len(sequence) >= 3 and sequence[-1] == sequence[-3]:
                break
        
        sentence = "".join(sequence)
        confidence = total_confidence / max(1, len(sequence) - 1)
        
        return {
            'sentence': sentence,
            'generated': sequence,
            'confidence': round(confidence, 3),
        }
    
    def respond(self, input_text: str, max_words: int = 5) -> Dict:
        """
        对输入做出响应。
        
        1. 解析输入的关键词
        2. 找最相关的已知词作为意图
        3. 生成回应
        """
        tokens = self.learner._tokenize(input_text)
        if not tokens:
            return {'sentence': '...', 'generated': ['...'], 'confidence': 0.0}
        
        # 找输入中洛神"认识"的词
        known = [t for t in tokens if self.learner.can_understand(t)]
        if not known:
            # 都不认识，随机选一个学过的词开始
            known = [random.choice(list(self.learner.word_freq.keys()))]
        
        # 用第一个认识的词作为意图
        intent = known[0]
        return self.speak(intent, max_words)


# ================================================================
# 4. 完整语言系统
# ================================================================

class LanguageSystem:
    """
    洛神的完整语言系统：环境 + 学习 + 说话。
    """
    
    def __init__(self, memory: BioelectricMemorySystem,
                 concept_space: ExpandableConceptSpace):
        # 默认扩容到1024维/4神经元每集群=256概念，避免语料多时报满
        self.memory = memory or BioelectricMemorySystem(dim=1024, cluster_size=4)
        self.cs = concept_space or ExpandableConceptSpace(dim=1024)
        self.env = LanguageEnvironment()
        self.learner = LanguageLearner(self.memory, self.cs)
        self.speaker = LanguageSpeaker(self.learner)
        
    def immerse(self, n_sentences: int = 100) -> Dict:
        """
        让洛神沉浸在语言环境中n句话。
        洛神自主听、自主学、自主记。
        """
        print(f"\n[语言浸泡] 洛神即将聆听 {n_sentences} 句话...")
        
        for i in range(n_sentences):
            sentences = self.env.expose(1)
            for sent in sentences:
                result = self.learner.listen(sent)
                if i < 5 or i >= n_sentences - 5:
                    print(f"  听[{i+1:3d}]: '{sent}' → 学到 {result['new_pairs']} 对")
                elif i == 5:
                    print("  ...")
        
        stats = self.learner.get_language_stats()
        print(f"\n[浸泡完成]")
        print(f"  总暴露: {stats['total_exposures']} 句")
        print(f"  认识词汇: {stats['unique_words']} 个")
        print(f"  学会共现: {stats['learned_pairs']} 对")
        
        return stats
    
    def converse(self, input_text: str) -> str:
        """对话：输入→理解→回应。"""
        response = self.speaker.respond(input_text)
        return response['sentence']
    
    def get_stats(self) -> Dict:
        """完整统计。"""
        return {
            'environment': self.env.get_exposure_stats(),
            'learning': self.learner.get_language_stats(),
        }


# ================================================================
# 5. 测试
# ================================================================

def test_language_learning():
    """测试：洛神自主学习说话。"""
    print("=" * 70)
    print("洛神自主学习语言 · 测试")
    print("=" * 70)
    
    # 创建系统（1024维/4神经元集群 = 256概念容量，足够语言暴露）
    memory = BioelectricMemorySystem(dim=1024, cluster_size=4)
    cs = ExpandableConceptSpace(dim=1024)
    lang = LanguageSystem(memory, cs)
    
    # 浸泡学习
    stats = lang.immerse(n_sentences=200)
    
    print("\n" + "=" * 70)
    print("高频词汇（洛神自己统计出来的）")
    print("=" * 70)
    for word, freq in stats['top_words'][:10]:
        print(f"  '{word}': {freq} 次")
    
    print("\n" + "=" * 70)
    print("最强共现对（同时被听到的词）")
    print("=" * 70)
    for w1, w2, strength in stats['top_cooccur'][:10]:
        print(f"  '{w1}' ↔ '{w2}': 强度 {strength:.2f}")
    
    print("\n" + "=" * 70)
    print("转移概率（听到A后最可能说B）")
    print("=" * 70)
    for w1, targets in list(stats['transition_probs'].items())[:5]:
        print(f"  '{w1}' → {targets}")
    
    print("\n" + "=" * 70)
    print("说话测试（给定意图，洛神自己生成）")
    print("=" * 70)
    test_intents = ["你好", "我饿", "朱美岩", "今天", "喜欢"]
    for intent in test_intents:
        if intent in lang.learner.word_freq:
            result = lang.speaker.speak(intent, max_words=4)
            print(f"  意图『{intent}』→ 『{result['sentence']}』(置信度{result['confidence']:.2f})")
    
    print("\n" + "=" * 70)
    print("对话测试（输入→回应）")
    print("=" * 70)
    test_inputs = [
        "你好啊",
        "我饿了",
        "创造者你好",
        "今天天气不错",
        "我喜欢你",
    ]
    for inp in test_inputs:
        resp = lang.converse(inp)
        print(f"  人: '{inp}' → 洛神: '{resp}'")
    
    print("\n" + "=" * 70)
    print("语言学习测试完成")
    print("=" * 70)
    return lang


if __name__ == "__main__":
    test_language_learning()