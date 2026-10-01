#!/usr/bin/env python3
"""
curiosity_learning_loop.py — 好奇心学习闭环
============================================

把 question_engine + knowledge_feeder + 生物学生长引擎 整合，
实现完整的"遇到不懂→提问→学习→回应"循环。

核心流程：
1. 用户输入
2. question_engine 分析：有没有未知概念？
3. 如果有 → KnowledgeFeeder 自动学习（百度百科→儿童语言→注入记忆）
4. 同时生成提问回应："<概念>是什么呀？"
5. 用户解释后 → 再次学习 → 生成新回应

生物学原理：
- 好奇心驱动（内在动机）
- 错误驱动学习（遇到未知→学习）
- 强化学习（用户回应=奖励信号）
"""

import sys
import os
sys.path.insert(0, '01_environment')

from language_learner_v2 import LanguageSystemV2
from question_engine import QuestionEngine
from knowledge_feeder import KnowledgeFeeder
from authentic_expression_engine import AuthenticExpressionEngine, BioelectricSnapshot
from biological_growth import NeuroplasticConceptMapping, LanguageDevelopmentTracker


class CuriosityLearningLoop:
    """
    好奇心学习闭环。
    
    洛神的核心能力：不是只会回答，是遇到不懂的会主动问、主动学。
    """
    
    def __init__(self, language_system):
        self.lang = language_system
        self.question_engine = QuestionEngine(language_system)
        self.knowledge_feeder = KnowledgeFeeder(language_system)
        self.expression_engine = AuthenticExpressionEngine(language_system)
        
        # 学习统计
        self.learned_from_questions = 0
        self.unknown_concepts_history = []
        
        print("[好奇心闭环] 初始化完成")
    
    def process(self, user_input: str, user_emotion: dict = None) -> dict:
        """
        处理用户输入，可能触发学习和提问。
        
        返回：
        {
            'response': ['句子1', '句子2'],
            'action': 'question' | 'respond' | 'learned_and_respond',
            'unknown_concepts': ['概念1'],
            'learned': True | False,
        }
        """
        # 1. 分词
        tokens = self.lang.tokenizer.tokenize(user_input)
        
        # 2. 分析是否需要提问
        analysis = self.question_engine.analyze_input(user_input, tokens)
        
        if analysis['should_ask']:
            # 检测到未知概念
            unknown_concept = analysis['target_concept']
            self.unknown_concepts_history.append(unknown_concept)
            
            # 自动学习（知识投喂）
            learned = self._auto_learn(unknown_concept)
            
            # 生成提问
            question = self.question_engine.generate_question(
                analysis, 
                user_emotion.get('dominant', '好奇') if user_emotion else '好奇'
            )
            
            # 生成情感回应
            emotion = user_emotion or {'dominant': '好奇', 'valence': 0.3, 'arousal': 0.5}
            snap = BioelectricSnapshot(
                emotion, unknown_concept, 0.85, 'now'
            )
            expression = self.expression_engine.express(snap)
            
            # 组合回应：提问 + 表达
            response = [question] + expression['monologue'][:1]
            
            return {
                'response': response,
                'action': 'learned_and_respond' if learned else 'question',
                'unknown_concepts': [unknown_concept],
                'learned': learned,
                'question': question,
            }
        
        # 不需要提问，正常回应
        return {
            'response': ['知道了'],
            'action': 'respond',
            'unknown_concepts': [],
            'learned': False,
        }
    
    def _auto_learn(self, concept: str) -> bool:
        """
        自动学习未知概念。
        
        尝试从知识库学习，如果没有则标记待学习。
        """
        # 先尝试从内置知识库学习
        result = self.knowledge_feeder.feed_topic(concept, max_sentences=10)
        
        if result['status'] == 'fed':
            self.learned_from_questions += 1
            print(f"[自动学习] 已学习'{concept}'：{result['n_sentences']}句")
            return True
        
        # 知识库没有这个概念，标记为待学习
        print(f"[待学习] '{concept}'不在知识库中，需要外部知识")
        return False
    
    def get_stats(self) -> dict:
        """获取好奇心学习统计。"""
        return {
            'learned_from_questions': self.learned_from_questions,
            'unknown_concepts_history': self.unknown_concepts_history,
            'knowledge_base_topics': len(self.knowledge_feeder.corpus.knowledge),
        }


def test_curiosity_loop():
    """测试好奇心学习闭环。"""
    print("=" * 60)
    print("好奇心学习闭环测试")
    print("=" * 60)
    
    # 初始化
    lang = LanguageSystemV2()
    lang.immerse(n_sentences=300)
    
    loop = CuriosityLearningLoop(lang)
    
    # 测试场景
    test_inputs = [
        "你知道黑洞吗",      # 未知概念：黑洞
        "太阳出来了",        # 已知概念
        "量子力学很有趣",    # 未知概念：量子力学
        "花儿开了",          # 已知概念
    ]
    
    print("\n模拟对话：")
    for user_input in test_inputs:
        print(f"\n[用户] {user_input}")
        result = loop.process(user_input)
        print(f"[洛神] {' '.join(result['response'])}")
        print(f"  动作: {result['action']}")
        if result['unknown_concepts']:
            print(f"  未知概念: {result['unknown_concepts']}")
            print(f"  已学习: {result['learned']}")
    
    # 统计
    print("\n" + "=" * 60)
    stats = loop.get_stats()
    print(f"从提问中学习: {stats['learned_from_questions']}次")
    print(f"未知概念历史: {stats['unknown_concepts_history']}")
    print("=" * 60)


if __name__ == '__main__':
    test_curiosity_loop()
