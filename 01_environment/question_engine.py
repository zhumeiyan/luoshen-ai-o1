#!/usr/bin/env python3
"""
question_engine.py — 自主提问引擎
==========================

让洛神在合适的时候主动提问，而不是只能被动回应。

核心功能：
1. 未知概念检测：遇到没学过的词时，主动问"这是什么？"
2. 上下文断裂检测：当用户话题跳转太大时，主动确认
3. 好奇心驱动提问：对感兴趣但不了解的概念追问细节
4. 情感化提问风格：不同情感下提问方式不同（开心时好奇，难过时求助）

提问策略：
- 识别到未知词 → "<词>是什么呀？"
- 话题突然跳转 → "你是说<新话题>吗？"
- 上下文不连贯 → "然后呢？"
- 对已知概念的延伸好奇 → "<概念>为什么会<属性>？"

与6-7岁儿童行为对齐：
- 爱问"为什么""是什么"
- 对新鲜事物好奇
- 不确定时会确认
- 难过时寻求解释/安慰
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random
import sys

sys.path.insert(0, '01_environment')
from complex_syntax_engine import ComplexSyntaxEngine, ComplexPattern


class QuestionEngine:
    """
    自主提问引擎。
    
    让洛神学会在合适的时候提出合适的问题。
    """
    
    def __init__(self, language_system):
        self.lang = language_system
        self.syntax = ComplexSyntaxEngine(language_system)
        
        # 提问模板库
        self.question_templates = self._build_templates()
        
        # 已知概念阈值：频次>=2才算"学会"
        self.known_threshold = 2
        
        # 提问冷却：同一概念短时间内不问重复问题
        self.questioned_recently = set()
        self.max_recent = 20
        
        # 连续提问计数（避免话痨）
        self.consecutive_questions = 0
        self.max_consecutive = 2
    
    def _build_templates(self) -> Dict[str, List[str]]:
        """构建提问模板库。"""
        return {
            'unknown_concept': [  # 未知概念
                "{concept}是什么呀？",
                "什么是{concept}？",
                "{concept}是做什么的？",
                "我不懂{concept}...",
            ],
            'topic_switch': [  # 话题跳转确认
                "你是说{concept}吗？",
                "我们要聊{concept}了吗？",
                "{concept}？这是什么意思？",
            ],
            'curiosity': [  # 好奇心追问
                "{concept}为什么会{attribute}？",
                "{concept}是怎么{attribute}的？",
                "为什么{concept}{attribute}？",
                "{concept}还有别的{attribute}吗？",
            ],
            'confirm': [  # 确认理解
                "你是说{sentence}吗？",
                "我理解对了吗？{sentence}",
                "所以是{sentence}？",
            ],
            'emotion_driven': {  # 情感驱动提问
                '开心': [
                    "{concept}好玩吗？",
                    "{concept}有趣吗？",
                    "我们可以{concept}吗？",
                ],
                '难过': [
                    "{concept}会让我开心吗？",
                    "{concept}能帮我吗？",
                    "为什么{concept}？",
                ],
                '害怕': [
                    "{concept}危险吗？",
                    "{concept}会伤害我吗？",
                    "{concept}安全吗？",
                ],
                '好奇': [
                    "{concept}是什么做成的？",
                    "{concept}从哪里来？",
                    "{concept}为什么会这样？",
                ],
                '平静': [
                    "{concept}是什么？",
                    "{concept}在哪里？",
                ],
            },
            'context_gap': [  # 上下文断裂
                "然后呢？",
                "接下来呢？",
                "后来怎么样了？",
                "还有呢？",
            ],
        }
    
    def analyze_input(self, user_input: str, user_concepts: List[str]) -> Dict:
        """
        分析用户输入，判断是否需要提问以及问什么。
        
        返回：
            {
                'should_ask': bool,
                'question_type': str,
                'target_concept': str,
                'reason': str,
            }
        """
        # 1. 检查未知概念
        unknown = self._find_unknown_concepts(user_concepts)
        if unknown and self.consecutive_questions < self.max_consecutive:
            return {
                'should_ask': True,
                'question_type': 'unknown_concept',
                'target_concept': unknown[0],
                'reason': f"遇到未知概念: {unknown[0]}",
            }
        
        # 2. 检查上下文断裂（如果提供了上下文信息，由调用方传入）
        # 这里只做简单检测：输入极短且没有已知概念
        known_count = sum(1 for c in user_concepts if self._is_known(c))
        if len(user_concepts) >= 2 and known_count == 0:
            return {
                'should_ask': True,
                'question_type': 'topic_switch',
                'target_concept': user_concepts[0] if user_concepts else '这个',
                'reason': "话题跳转，无已知概念",
            }
        
        # 3. 好奇心驱动（随机但基于兴趣）
        # 对已知但不太熟悉的概念（频次2-5）有好奇心
        semi_known = [c for c in user_concepts if self._is_semi_known(c)]
        if semi_known and random.random() < 0.3 and self.consecutive_questions < self.max_consecutive:
            target = random.choice(semi_known)
            return {
                'should_ask': True,
                'question_type': 'curiosity',
                'target_concept': target,
                'reason': f"对半熟悉概念好奇: {target}",
            }
        
        # 默认不提问
        return {'should_ask': False, 'question_type': None, 'target_concept': None, 'reason': '无需提问'}
    
    def generate_question(self, analysis: Dict, emotion: str = '平静') -> str:
        """
        根据分析结果生成具体问题。
        
        Args:
            analysis: analyze_input() 的返回结果
            emotion: 当前情感状态
        """
        q_type = analysis['question_type']
        concept = analysis['target_concept']
        
        if not q_type or not concept:
            return ""
        
        # 记录提问
        self.questioned_recently.add(concept)
        if len(self.questioned_recently) > self.max_recent:
            self.questioned_recently.pop()
        self.consecutive_questions += 1
        
        # 选择模板
        templates = self.question_templates.get(q_type, [])
        
        # 情感驱动提问优先
        if q_type == 'curiosity' and emotion in self.question_templates['emotion_driven']:
            emo_templates = self.question_templates['emotion_driven'][emotion]
            if emo_templates:
                template = random.choice(emo_templates)
                return template.format(concept=concept, attribute="这样")
        
        if not templates:
            return f"{concept}是什么？"
        
        # 填充模板
        template = random.choice(templates)
        
        if q_type == 'curiosity':
            # 尝试找一个属性（从概念关联中）
            attribute = self._guess_attribute(concept)
            return template.format(concept=concept, attribute=attribute)
        
        return template.format(concept=concept, sentence=f"关于{concept}")
    
    def reset_consecutive(self):
        """当收到用户回应后，重置连续提问计数。"""
        self.consecutive_questions = 0
    
    def _find_unknown_concepts(self, concepts: List[str]) -> List[str]:
        """找出概念列表中的未知概念。"""
        unknown = []
        for c in concepts:
            if not self._is_known(c) and c not in self.questioned_recently:
                unknown.append(c)
        return unknown
    
    def _is_known(self, concept: str) -> bool:
        """判断概念是否已学会。"""
        if not hasattr(self.lang, 'learned_words'):
            return False
        freq = self.lang.learned_words.get(concept, 0)
        return freq >= self.known_threshold
    
    def _is_semi_known(self, concept: str) -> bool:
        """判断概念是否半熟悉（学过但不深）。"""
        if not hasattr(self.lang, 'learned_words'):
            return False
        freq = self.lang.learned_words.get(concept, 0)
        return 1 <= freq < 5  # 学过但频次不高
    
    def _guess_attribute(self, concept: str) -> str:
        """猜测概念的一个属性（用于好奇提问）。"""
        # 从G矩阵关联中找一个相关概念作为属性
        if hasattr(self.lang, 'memory'):
            try:
                recall = self.lang.memory.recall(concept, steps=5)
                activated = recall.get('activated_concepts', {})
                if activated:
                    # 取激活度最高的相关概念
                    best = max(activated.items(), key=lambda x: x[1])[0]
                    if best != concept:
                        return best
            except:
                pass
        return "这样"


def test_question_engine():
    """测试自主提问引擎。"""
    print("=" * 60)
    print("自主提问引擎测试")
    print("=" * 60)
    
    from language_learner_v2 import LanguageSystemV2
    
    lang = LanguageSystemV2()
    lang.immerse(n_sentences=300)
    
    engine = QuestionEngine(lang)
    
    # 模拟对话场景
    test_scenarios = [
        {
            'input': '你听说过量子力学吗',
            'concepts': ['你', '听说', '过', '量子', '力学'],
            'emotion': '好奇',
        },
        {
            'input': '今天天气真好',
            'concepts': ['今天', '天气', '真', '好'],
            'emotion': '开心',
        },
        {
            'input': '那个东西',
            'concepts': ['那个', '东西'],
            'emotion': '平静',
        },
        {
            'input': '我害怕黑洞',
            'concepts': ['我', '害怕', '黑洞'],
            'emotion': '害怕',
        },
    ]
    
    print(f"\n当前词汇量: {len(lang.learned_words)}")
    print("高频词样例:", sorted(lang.learned_words.items(), key=lambda x: -x[1])[:10])
    
    print("\n" + "-" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n场景{i}: [{scenario['emotion']}] {scenario['input']}")
        
        analysis = engine.analyze_input(scenario['input'], scenario['concepts'])
        print(f"  分析: {analysis['reason']}")
        
        if analysis['should_ask']:
            question = engine.generate_question(analysis, scenario['emotion'])
            print(f"  提问: {question}")
        else:
            print(f"  提问: (不提问)")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_question_engine()
