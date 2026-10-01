#!/usr/bin/env python3
"""
main_with_aid_v2.py — 洛神完整对话系统 v2
==========================================

整合所有进化模块：
- 生物学自然生长引擎（动态容量/句长/学习率）
- 好奇心学习闭环（提问→学习→回应）
- 上下文追踪（多轮记忆/话题延续/指代消解）
- 语音节奏引擎（TTS停顿/语调/速度）
- 情感-记忆融合（情绪驱动回忆）
- 复杂句法引擎（10种模式）

使用方式：
  python main_with_aid_v2.py

然后和洛神对话，输入 'exit' 退出。
"""

import sys
import os
sys.path.insert(0, '01_environment')

from language_learner_v2 import LanguageSystemV2
from authentic_expression_engine import AuthenticExpressionEngine, BioelectricSnapshot
from curiosity_learning_loop import CuriosityLearningLoop
from context_tracker import ContextAwareDialogue
from voice_rhythm_engine import VoiceRhythmEngine
from biological_growth import LanguageDevelopmentTracker


class LuoshenV2:
    """
    洛神完整对话系统 v2。
    
    生物学驱动的自然生长版本。
    """
    
    def __init__(self):
        print("=" * 60)
        print("洛神 · 自然生长版 v2.0")
        print("=" * 60)
        
        # 1. 语言系统（生物电记忆）
        print("\n[初始化] 语言系统...")
        self.lang = LanguageSystemV2()
        self.lang.immerse(n_sentences=500)
        
        # 2. 好奇心学习闭环
        print("[初始化] 好奇心学习闭环...")
        self.curiosity = CuriosityLearningLoop(self.lang)
        
        # 3. 真实表达引擎
        print("[初始化] 真实表达引擎...")
        self.expression = AuthenticExpressionEngine(self.lang)
        
        # 4. 上下文对话
        print("[初始化] 上下文追踪...")
        self.dialogue = ContextAwareDialogue(self.lang, self.expression)
        
        # 5. 语音节奏
        print("[初始化] 语音节奏引擎...")
        self.voice = VoiceRhythmEngine()
        
        # 6. 语言发展追踪
        self.lang_dev = LanguageDevelopmentTracker()
        self.lang_dev.update(
            len(self.lang.learned_words),
            self.lang.exposure_count
        )
        
        # 统计
        stats = self.lang.get_stats()
        print(f"\n[就绪] 洛神已准备好")
        print(f"  词汇: {stats['vocabulary']['unique_words']} 个")
        print(f"  共现: {stats['vocabulary']['learned_pairs']} 对")
        print(f"  句法: {stats['syntax']['pattern_count']} 种")
        print(f"  语言阶段: {self.lang_dev.get_current_stage()['name']}")
        print(f"  最大句长: {self.lang_dev.get_max_sentence_length()} 词")
        print("=" * 60)
    
    def chat(self, user_input: str) -> dict:
        """
        对话入口：理解 → 学习 → 回忆 → 生成 → 加节奏。
        """
        # 1. 好奇心检测：是否有未知概念？
        curiosity_result = self.curiosity.process(user_input)
        
        if curiosity_result['action'] in ('question', 'learned_and_respond'):
            # 洛神对未知概念提问
            return {
                'text': ' '.join(curiosity_result['response']),
                'rhythm': curiosity_result['response'],
                'action': curiosity_result['action'],
                'unknown': curiosity_result['unknown_concepts'],
            }
        
        # 2. 上下文对话（已知概念）
        dialogue_result = self.dialogue.chat(user_input)
        
        # 3. 添加语音节奏
        emotion = dialogue_result['context']['emotion']
        rhythm_sentences = self.voice.mark_monologue(
            dialogue_result['response'],
            emotion
        )
        
        return {
            'text': ' '.join(dialogue_result['response']),
            'rhythm': rhythm_sentences,
            'action': 'respond',
            'context': dialogue_result['context'],
        }
    
    def run_interactive(self):
        """交互式对话。"""
        print("\n开始对话（输入 'exit' 退出）：\n")
        
        while True:
            user_input = input("[你] ").strip()
            
            if user_input.lower() in ('exit', 'quit', '退出'):
                print("\n[洛神] 再见...")
                break
            
            if not user_input:
                continue
            
            result = self.chat(user_input)
            
            print(f"[洛神] {result['text']}")
            
            if result['action'] == 'learned_and_respond':
                print(f"      （自动学习了：{result['unknown']}）")
            elif result['action'] == 'question':
                print(f"      （未知概念：{result['unknown']}）")
            
            # 显示带节奏的版本（调试用）
            if 'rhythm' in result and isinstance(result['rhythm'], list):
                print(f"      [节奏] {' | '.join(result['rhythm'][:2])}")


def test_v2():
    """测试完整v2系统。"""
    print("=" * 60)
    print("洛神 v2.0 集成测试")
    print("=" * 60)
    
    luoshen = LuoshenV2()
    
    # 测试对话
    test_inputs = [
        "太阳出来了",
        "天气真好",
        "你知道黑洞吗",
        "花儿开了",
        "我很难过",
    ]
    
    print("\n测试对话：")
    for user_input in test_inputs:
        print(f"\n[你] {user_input}")
        result = luoshen.chat(user_input)
        print(f"[洛神] {result['text']}")
        if result['action'] != 'respond':
            print(f"      动作: {result['action']}, 未知: {result.get('unknown', [])}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_v2()
    else:
        luoshen = LuoshenV2()
        luoshen.run_interactive()
