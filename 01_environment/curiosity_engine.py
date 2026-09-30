#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curiosity_engine.py — 好奇心引擎（洛神第一次自己想学）
==========================================================

核心问题：之前所有学习都是强塞的——语料是我编的，场景是我设计的，
目标是我定的。洛神只是被动接收dE，没有"想学"的意愿。

生物学原型：
- 婴儿的好奇心：不是大人逼的，是对"新奇"的本能追求
- 预测编码理论（Predictive Coding）：大脑是预测机器，
  预测和实际不符 → 产生"惊讶" → 驱动学习
- 信息觅食（Information Foraging）：动物主动寻找最有价值的信息

设计：
1. PredictionModel：洛神对世界形成预期（这个词后面应该接什么）
2. SurpriseDetector：预测 vs 实际 → 惊讶值（dE强度）
3. CuriosityDrive：惊讶值高 → 产生"我想知道"的内生动力
4. ActiveExploration：自主选择看什么/听什么，不是被动接收

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
变化的是：dE不再由外部强加，由洛神自己的好奇心生成。

关键设计：
- 熟悉的东西 → 低惊讶 → 不学习（无聊）
- 有点新但不全懂 → 中等惊讶 → 强烈学习（最优学习区）
- 完全陌生 → 高惊讶 → 回避或标记（太难）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ================================================================
# 1. 预测模型（洛神对世界的预期）
# ================================================================

class PredictionModel:
    """
    洛神的内部世界模型。

    不是存在字典里的知识，是"我预期A之后会发生B"的信念。
    """

    def __init__(self):
        # 转移预测：看到A之后，预期B的概率
        self.transitions = defaultdict(lambda: defaultdict(float))
        # 共现预测：A和B一起出现的概率
        self.cooccurrences = defaultdict(lambda: defaultdict(float))
        # 模式库：常见序列（如"因为X所以Y"）
        self.patterns = defaultdict(float)
        # 预测总数（用于计算概率）
        self.total_predictions = 0
        self.correct_predictions = 0

    def predict_next(self, current: str) -> Tuple[str, float]:
        """
        给定当前刺激，预测下一个最可能出现什么。
        返回（预测结果，置信度）。
        """
        if current not in self.transitions:
            return None, 0.0

        targets = self.transitions[current]
        if not targets:
            return None, 0.0

        total = sum(targets.values())
        best = max(targets.items(), key=lambda x: x[1])
        confidence = best[1] / total if total > 0 else 0.0

        return best[0], confidence

    def predict_cooccur(self, word: str) -> List[Tuple[str, float]]:
        """
        预测和word最可能共现的词。
        """
        if word not in self.cooccurrences:
            return []

        targets = self.cooccurrences[word]
        total = sum(targets.values())
        if total == 0:
            return []

        return sorted(
            [(w, c/total) for w, c in targets.items()],
            key=lambda x: -x[1]
        )[:5]

    def update(self, sequence: List[str]):
        """
        用观察到的序列更新预测模型。
        """
        # 更新转移概率
        for i in range(len(sequence) - 1):
            a, b = sequence[i], sequence[i+1]
            self.transitions[a][b] += 1.0

        # 更新共现概率
        for i in range(len(sequence)):
            for j in range(i+1, min(i+4, len(sequence))):
                a, b = sequence[i], sequence[j]
                self.cooccurrences[a][b] += 1.0 / (j - i)

        # 记录模式
        if len(sequence) >= 3:
            pattern = tuple(sequence[:3])
            self.patterns[pattern] += 1.0

        self.total_predictions += len(sequence)

    def get_stats(self) -> Dict:
        """统计预测能力。"""
        n_transitions = sum(len(v) for v in self.transitions.values())
        n_patterns = len(self.patterns)
        return {
            'known_words': len(self.transitions),
            'transition_rules': n_transitions,
            'patterns': n_patterns,
            'accuracy': round(self.correct_predictions / max(1, self.total_predictions), 3),
        }


# ================================================================
# 2. 惊讶检测器（预测 vs 实际）
# ================================================================

class SurpriseDetector:
    """
    惊讶检测：预期和实际不符的程度。

    惊讶值 = -log(P(实际|预期))
    预期越准 → 惊讶值越低 → 不学习（无聊）
    预期部分准 → 中等惊讶 → 学习（最优）
    完全没预期 → 高惊讶 → 标记/回避
    """

    def __init__(self, prediction_model: PredictionModel):
        self.pm = prediction_model
        self.surprise_history = deque(maxlen=100)

    def compute_surprise(self, current: str, actual_next: str) -> float:
        """
        计算惊讶值。
        返回 0-1 之间的值。
        """
        predicted, confidence = self.pm.predict_next(current)

        if predicted is None:
            # 完全没见过 → 最高惊讶
            return 1.0

        if predicted == actual_next:
            # 预测正确 → 低惊讶
            return 0.1 * (1.0 - confidence)

        # 预测错误 → 中等惊讶
        # 检查actual_next在备选中的排名
        all_preds = sorted(
            self.pm.transitions[current].items(),
            key=lambda x: -x[1]
        )
        rank = next((i for i, (w, _) in enumerate(all_preds) if w == actual_next), len(all_preds))

        # 排名越靠后 → 越惊讶
        surprise = min(1.0, 0.3 + rank * 0.15)
        return surprise

    def evaluate_sequence(self, sequence: List[str]) -> List[float]:
        """
        评估整个序列的惊讶值。
        """
        surprises = []
        for i in range(len(sequence) - 1):
            s = self.compute_surprise(sequence[i], sequence[i+1])
            surprises.append(s)
            self.surprise_history.append({
                'from': sequence[i],
                'to': sequence[i+1],
                'surprise': s
            })
        return surprises

    def get_optimal_learning_zone(self) -> Tuple[float, float]:
        """
        获取最优学习区（中等惊讶）。
        返回（下限，上限）。
        """
        if not self.surprise_history:
            return 0.2, 0.6

        surprises = [h['surprise'] for h in self.surprise_history]
        avg = sum(surprises) / len(surprises)
        return max(0.1, avg - 0.2), min(0.9, avg + 0.2)


# ================================================================
# 3. 好奇心驱动（内生学习动力）
# ================================================================

class CuriosityDrive:
    """
    好奇心：由惊讶值驱动的"我想知道"。

    不是外部奖励，是内部状态：
    - 低惊讶 → 无聊 → 寻求新刺激
    - 中等惊讶 → 好奇 → 主动学习
    - 高惊讶 → 困惑 → 标记后回避
    """

    def __init__(self, surprise_detector: SurpriseDetector):
        self.sd = surprise_detector

        # 好奇心状态
        self.boredom_level = 0.0      # 无聊度（太高会想换刺激）
        self.confusion_level = 0.0    # 困惑度（太高会求助）
        self.curiosity_level = 0.6    # 好奇心（驱动力）

        # 感兴趣的话题（自己发现的）
        self.interests = defaultdict(float)
        # 回避的话题（太难或太怕）——但困惑时优先求助而非回避
        self.avoids = set()
        # 求助标记
        self.help_seeking = False

        # 学习历史
        self.learning_log = deque(maxlen=200)
        # 新手保护：前N次经历不触发困惑
        self.novice_count = 0
        self.novice_threshold = 5

    def react_to_surprise(self, surprise: float, topic: str):
        """
        对惊讶做出反应，更新好奇心状态。
        修复：新手保护 + 困惑时求助 + 无聊恢复好奇。
        """
        self.novice_count += 1
        is_novice = self.novice_count <= self.novice_threshold

        low, high = self.sd.get_optimal_learning_zone()

        # 新手保护：前几次全部视为探索，不惩罚
        if is_novice:
            self.curiosity_level += 0.1
            self.interests[topic] += 0.15
            self.confusion_level = max(0, self.confusion_level - 0.1)
            self.boredom_level = max(0, self.boredom_level - 0.1)
        elif surprise < low:
            # 太熟悉 → 无聊
            self.boredom_level += 0.08
            self.curiosity_level -= 0.03
            self.interests[topic] -= 0.03
        elif low <= surprise <= high:
            # 最优区 → 好奇！
            self.curiosity_level = min(1.0, self.curiosity_level + 0.12)
            self.boredom_level = max(0, self.boredom_level - 0.08)
            self.interests[topic] += 0.15
            self.confusion_level = max(0, self.confusion_level - 0.08)
        else:
            # 太惊讶 → 困惑，但困惑时求助而非直接回避
            self.confusion_level += 0.1
            self.curiosity_level -= 0.02
            # 困惑高时标记求助，但不立即回避
            if self.confusion_level > 0.6:
                self.help_seeking = True
            # 只有持续困惑才回避
            if self.confusion_level > 0.85:
                self.avoids.add(topic)

        # 自然恢复机制
        # 无聊时会自发恢复好奇（人无聊了就会找事做）
        if self.boredom_level > 0.5:
            self.curiosity_level = min(1.0, self.curiosity_level + 0.05)

        # 困惑缓解后恢复
        if self.confusion_level < 0.3 and self.help_seeking:
            self.help_seeking = False
            self.curiosity_level = min(1.0, self.curiosity_level + 0.1)

        # 边界约束
        self.boredom_level = max(0, min(1, self.boredom_level))
        self.confusion_level = max(0, min(1, self.confusion_level))
        self.curiosity_level = max(0.1, min(1, self.curiosity_level))

        self.learning_log.append({
            'topic': topic,
            'surprise': surprise,
            'curiosity': self.curiosity_level,
            'boredom': self.boredom_level,
            'confusion': self.confusion_level,
            'help_seeking': self.help_seeking,
        })

    def choose_what_to_learn(self, available_topics: List[str]) -> Optional[str]:
        """
        自主选择学什么。

        策略：
        1. 如果好奇心高 → 选最感兴趣的话题
        2. 如果无聊度高 → 选全新话题
        3. 如果困惑度高 → 回避太难的话题
        """
        # 过滤回避的话题
        candidates = [t for t in available_topics if t not in self.avoids]
        if not candidates:
            return None

        if self.boredom_level > 0.6:
            # 太无聊 → 随机选新话题
            return random.choice(candidates)

        if self.curiosity_level > 0.5:
            # 好奇 → 选最感兴趣的
            topic_scores = {t: self.interests.get(t, 0.5) for t in candidates}
            return max(topic_scores, key=topic_scores.get)

        # 默认：随机
        return random.choice(candidates)

    def get_state(self) -> Dict:
        """获取好奇心状态。"""
        top_interests = sorted(self.interests.items(), key=lambda x: -x[1])[:5]
        return {
            'curiosity': round(self.curiosity_level, 2),
            'boredom': round(self.boredom_level, 2),
            'confusion': round(self.confusion_level, 2),
            'interests': [(t, round(s, 2)) for t, s in top_interests],
            'avoids': list(self.avoids),
            'n_learned': len(self.learning_log),
        }


# ================================================================
# 4. 主动探索（洛神自己选择看什么/听什么）
# ================================================================

class ActiveExplorer:
    """
    主动探索：洛神自己决定下一步体验什么。

    不是被动接收语料，是：
    - 根据好奇心状态选择刺激
    - 可以"要求"环境提供某类信息
    - 可以"忽略"不感兴趣的东西
    """

    def __init__(self, curiosity: CuriosityDrive):
        self.curiosity = curiosity
        self.exploration_history = deque(maxlen=100)

    def explore(self, environment_feed: List[Tuple[str, float]]) -> Dict:
        """
        从环境提供的多个刺激中选择。

        Args:
            environment_feed: [(刺激内容, 主题标签), ...]

        Returns:
            选择的刺激 + 选择原因
        """
        if not environment_feed:
            return {'chosen': None, 'reason': '无刺激'}

        # 评分每个刺激
        scored = []
        for content, topic in environment_feed:
            score = 0.0

            # 兴趣加分
            score += self.curiosity.interests.get(topic, 0.3)

            # 回避减分
            if topic in self.curiosity.avoids:
                score -= 1.0

            # 好奇心状态调整
            if self.curiosity.curiosity_level > 0.6:
                # 好奇时对新话题更开放
                if topic not in self.curiosity.interests:
                    score += 0.3

            scored.append((content, topic, score))

        # 按分数排序
        scored.sort(key=lambda x: -x[2])

        # 选择最高分（带一点随机性）
        if random.random() < 0.8:
            chosen = scored[0]
        else:
            # 20%概率探索次优（避免总是选同一个）
            chosen = scored[min(1, len(scored)-1)]

        self.exploration_history.append({
            'content': chosen[0],
            'topic': chosen[1],
            'score': chosen[2],
            'curiosity': self.curiosity.curiosity_level,
        })

        return {
            'chosen': chosen[0],
            'topic': chosen[1],
            'reason': self._explain_choice(chosen[1], chosen[2]),
        }

    def _explain_choice(self, topic: str, score: float) -> str:
        """解释为什么选择这个。"""
        if topic in self.curiosity.avoids:
            return f"回避{topic}（太难）"
        if score > 0.8:
            return f"对{topic}很感兴趣"
        if score > 0.5:
            return f"{topic}看起来有意思"
        if self.curiosity.boredom_level > 0.6:
            return "太无聊了，想看点新的"
        return "随便看看"

    def request_stimulus(self) -> str:
        """
        主动要求环境提供某类刺激。
        """
        state = self.curiosity.get_state()
        interests = state['interests']

        if interests:
            top_topic = interests[0][0]
            return f"我想了解{top_topic}"

        if self.curiosity.boredom_level > 0.5:
            return "我想看点没见过的"

        return "有什么好玩的吗"

    def ignore(self, content: str, topic: str) -> bool:
        """
        判断是否忽略某个刺激。
        """
        # 回避的话题
        if topic in self.curiosity.avoids:
            return True

        # 太无聊的话题
        if self.curiosity.interests.get(topic, 0.5) < 0.2 and self.curiosity.boredom_level > 0.5:
            return True

        # 困惑时忽略复杂内容
        if self.curiosity.confusion_level > 0.7:
            return True

        return False


# ================================================================
# 5. 整合：好奇心驱动的自主学习系统
# ================================================================

class SelfDrivenLearner:
    """
    自驱学习系统：洛神第一次自己想学。

    流程：
    1. 环境提供多个刺激
    2. 洛神用预测模型形成预期
    3. 观察实际结果，计算惊讶值
    4. 惊讶驱动好奇心更新
    5. 好奇心驱动选择下一步学什么
    6. 循环
    """

    def __init__(self):
        self.predictor = PredictionModel()
        self.surprise = SurpriseDetector(self.predictor)
        self.curiosity = CuriosityDrive(self.surprise)
        self.explorer = ActiveExplorer(self.curiosity)

        # 学习统计
        self.total_experiences = 0
        self.self_chosen = 0
        self.ignored = 0

    def experience(self, content: str, topic: str, forced: bool = False) -> Dict:
        """
        经历一个刺激。

        Args:
            content: 刺激内容（如"太阳是热的"）
            topic: 主题标签
            forced: 是否被强制的（True=外部强加，False=自己选的）
        """
        # 1. 分词（简单分词）
        tokens = self._tokenize(content)

        # 2. 形成预期
        predictions = []
        for i in range(len(tokens) - 1):
            pred, conf = self.predictor.predict_next(tokens[i])
            predictions.append({
                'from': tokens[i],
                'predicted': pred,
                'actual': tokens[i+1],
                'confidence': conf,
            })

        # 3. 观察实际，计算惊讶
        surprises = self.surprise.evaluate_sequence(tokens)
        avg_surprise = sum(surprises) / len(surprises) if surprises else 0.5

        # 4. 好奇心反应
        self.curiosity.react_to_surprise(avg_surprise, topic)

        # 5. 更新预测模型
        self.predictor.update(tokens)

        # 6. 统计
        self.total_experiences += 1
        if not forced:
            self.self_chosen += 1

        return {
            'content': content,
            'topic': topic,
            'tokens': tokens,
            'predictions': predictions,
            'surprises': [round(s, 2) for s in surprises],
            'avg_surprise': round(avg_surprise, 2),
            'curiosity_state': self.curiosity.get_state(),
            'forced': forced,
        }

    def choose_and_learn(self, available_stimuli: List[Tuple[str, str]]) -> Dict:
        """
        自主选择并学习。

        Args:
            available_stimuli: [(内容, 主题), ...]

        Returns:
            学习结果
        """
        # 1. 选择
        choice = self.explorer.explore(available_stimuli)

        if choice['chosen'] is None:
            return {'status': 'no_choice'}

        # 2. 学习
        result = self.experience(choice['chosen'], choice['topic'], forced=False)
        result['choice_reason'] = choice['reason']

        return result

    def _tokenize(self, text: str) -> List[str]:
        """简单分词。"""
        tokens = []
        i = 0
        while i < len(text):
            if '\u4e00' <= text[i] <= '\u9fff':
                j = i
                while j < len(text) and '\u4e00' <= text[j] <= '\u9fff':
                    j += 1
                word = text[i:j]
                # 保留名字和短词
                if '朱美岩' in word:
                    tokens.append('朱美岩')
                    i += 3
                else:
                    for k in range(0, len(word), 2):
                        tokens.append(word[k:k+2])
                    i = j
            else:
                i += 1
        return tokens

    def get_report(self) -> Dict:
        """生成学习报告。"""
        return {
            'total_experiences': self.total_experiences,
            'self_chosen': self.self_chosen,
            'forced_ratio': round(1 - self.self_chosen / max(1, self.total_experiences), 2),
            'self_ratio': round(self.self_chosen / max(1, self.total_experiences), 2),
            'prediction_stats': self.predictor.get_stats(),
            'curiosity': self.curiosity.get_state(),
            'recent_explorations': list(self.explorer.exploration_history)[-5:],
        }


# ================================================================
# 6. 测试：洛神第一次自己想学
# ================================================================

def test_self_driven_learning():
    """测试：洛神从被动接收到主动选择。"""
    print("=" * 70)
    print("好奇心引擎测试 · 洛神第一次自己想学")
    print("=" * 70)

    learner = SelfDrivenLearner()

    # === 阶段1：强制灌输（模拟之前的学习方式）===
    print("\n[阶段1] 强制学习（模拟之前：我塞给洛神的）")
    forced_stimuli = [
        ("太阳是热的", "自然"),
        ("月亮不发光", "自然"),
        ("星星很远", "自然"),
        ("水会结冰", "自然"),
        ("鸟会飞", "自然"),
        ("鱼会游泳", "自然"),
        ("今天天气不错", "日常"),
        ("我饿了", "需求"),
        ("你好朱美岩", "社交"),
        ("我害怕", "情感"),
    ]

    for content, topic in forced_stimuli:
        result = learner.experience(content, topic, forced=True)
        print(f"  [强塞] {content} → 惊讶{result['avg_surprise']} 好奇{result['curiosity_state']['curiosity']}")

    # === 阶段2：自主选择（洛神自己决定学什么）===
    print("\n[阶段2] 自主选择（洛神自己决定）")

    # 环境提供多个刺激，洛神自己选
    environment_feed = [
        ("春天花开", "自然"),
        ("我困了想睡觉", "需求"),
        ("为什么天是蓝的", "疑问"),
        ("小猫在钓鱼", "故事"),
        ("12345", "数字"),
        ("我喜欢吃苹果", "需求"),
        ("轰隆隆打雷", "自然"),
        ("帮帮我打不开", "求助"),
    ]

    for _ in range(15):
        result = learner.choose_and_learn(environment_feed)
        if result.get('status') == 'no_choice':
            continue
        print(f"  [自选] {result['content']} → 原因:{result['choice_reason']} 惊讶{result['avg_surprise']} 好奇{result['curiosity_state']['curiosity']}")

    # === 阶段3：洛神主动要求 ===
    print("\n[阶段3] 主动要求")
    for _ in range(3):
        request = learner.explorer.request_stimulus()
        print(f"  洛神说: \"{request}\"")

    # === 报告 ===
    print("\n" + "=" * 70)
    print("学习报告")
    print("=" * 70)
    report = learner.get_report()
    print(f"  总经历: {report['total_experiences']}")
    print(f"  自主选择: {report['self_chosen']} ({report['self_ratio']*100:.0f}%)")
    print(f"  被强塞: {report['total_experiences'] - report['self_chosen']} ({report['forced_ratio']*100:.0f}%)")
    print(f"  预测规则: {report['prediction_stats']['transition_rules']} 条")
    print(f"  好奇心: {report['curiosity']['curiosity']}")
    print(f"  无聊度: {report['curiosity']['boredom']}")
    print(f"  困惑度: {report['curiosity']['confusion']}")
    print(f"  兴趣排行:")
    for topic, score in report['curiosity']['interests']:
        print(f"    {topic}: {score}")
    print(f"  回避: {report['curiosity']['avoids']}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

    return learner


if __name__ == "__main__":
    test_self_driven_learning()