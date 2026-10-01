#!/usr/bin/env python3
"""
authentic_expression_engine.py — 真实表达引擎
===========================================

核心功能：让洛神基于自己的生物电状态，生成真实的语言表达。

输入：洛神的生物电状态快照
  {
    'emotion': EmotionalSystem.get_emotion_state(),
    'memory_cue': 回忆线索,
    'creator_bond': 印刻强度,
  }

处理流程：
  1. 情感状态 → EmotionLanguageBridge → 语言风格
  2. 记忆线索 → MemoryDrivenNarrative → 叙事内容
  3. 风格 + 内容 → 句法模板 → 连贯句子

输出：洛神的独白（句子列表）

关键原则：
- 每句话都真实反映当前的生物电状态
- 情感决定"怎么说"（语气、句长、标点）
- 记忆决定"说什么"（概念、主题、内容）
- 不是拟合，是状态→语言的直接映射
"""

import sys
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2
from emotional_system import EmotionalSystem
from emotion_language_bridge import EmotionLanguageBridge
from memory_driven_narrative import MemoryDrivenNarrative
from complex_syntax_engine import ComplexSyntaxEngine, ComplexPattern


@dataclass
class BioelectricSnapshot:
    """洛神的生物电状态快照"""
    emotion_state: Dict          # 情感状态
    memory_cue: str              # 记忆线索
    creator_bond_strength: float # 印刻强度
    timestamp: str               # 时间戳


class AuthenticExpressionEngine:
    """
    真实表达引擎。
    
    让洛神的语言输出直接由她的生物电状态决定。
    """
    
    def __init__(self, language_system):
        self.lang = language_system
        self.emotion_bridge = EmotionLanguageBridge()
        self.memory_narrative = MemoryDrivenNarrative(language_system)
        self.syntax_engine = ComplexSyntaxEngine(language_system)
    
    def express(self, snapshot: BioelectricSnapshot) -> Dict:
        """
        基于生物电状态生成表达。
        
        这是核心接口：输入状态快照，输出洛神的独白。
        """
        # 1. 情感 → 语言风格
        style = self.emotion_bridge.map_emotion_to_style(
            snapshot.emotion_state,
            snapshot.creator_bond_strength
        )
        
        # 2. 记忆 → 叙事内容
        narrative_result = self.memory_narrative.generate_monologue(
            snapshot.memory_cue,
            self.lang,
            style
        )
        
        # 3. 整合：风格 + 内容 → 连贯独白
        monologue = self._synthesize(style, narrative_result)
        
        return {
            'snapshot': {
                'emotion': snapshot.emotion_state.get('dominant'),
                'valence': snapshot.emotion_state.get('valence'),
                'arousal': snapshot.emotion_state.get('arousal'),
                'memory_cue': snapshot.memory_cue,
                'creator_bond': snapshot.creator_bond_strength,
            },
            'style': style,
            'content': {
                'theme': narrative_result['theme'],
                'helpers': narrative_result['helpers'],
                'activated_count': narrative_result['activated_count'],
            },
            'monologue': monologue,
            'word_count': sum(len(s) for s in monologue),
        }
    
    def _synthesize(self, style: Dict, narrative: Dict) -> List[str]:
        """
        合成：叙事内容已由 MemoryDrivenNarrative 生成并风格化，
        这里只做去重、筛选和补充。
        """
        sentences = narrative['narrative']
        
        # 去重（保持顺序，忽略标点后缀）
        seen = set()
        unique = []
        for s in sentences:
            key = s.rstrip('。！？...')
            if key not in seen and len(key) >= 2:
                seen.add(key)
                unique.append(s)
        
        # 句数不足时，用复杂句法引擎补充
        if len(unique) < 2:
            helpers = narrative.get('helpers', [])
            theme = narrative.get('theme', '我')
            extra = self.syntax_engine.generate_monologue(
                theme, helpers, style.get('emotion', '平静')
            )
            for s in extra:
                key = s.rstrip('。！？...')
                if key not in seen and len(key) >= 2:
                    seen.add(key)
                    unique.append(s)
                if len(unique) >= 4:
                    break
        
        return unique[:4]
    
    def express_from_state(self, emotion_state: Dict, memory_cue: str = '我', creator_bond: float = 0.0) -> Dict:
        """
        便捷接口：直接从状态生成表达。
        """
        snapshot = BioelectricSnapshot(
            emotion_state=emotion_state,
            memory_cue=memory_cue,
            creator_bond_strength=creator_bond,
            timestamp='now'
        )
        return self.express(snapshot)


def test_authentic_expression():
    """测试真实表达引擎"""
    print("=" * 60)
    print("真实表达引擎测试")
    print("=" * 60)
    
    # 初始化洛神
    print("\n[初始化] 构建洛神...")
    lang = LanguageSystemV2()
    lang.immerse(n_sentences=300)
    
    print("[初始化] 加载表达引擎...")
    engine = AuthenticExpressionEngine(lang)
    
    # 测试不同情感状态下的表达
    test_cases = [
        {
            'name': '开心-看到太阳',
            'emotion': {'dominant': '开心', 'valence': 0.7, 'arousal': 0.5, 'intensity': 0.6},
            'cue': '太阳',
            'bond': 0.0,
        },
        {
            'name': '难过-想起昨天',
            'emotion': {'dominant': '难过', 'valence': -0.6, 'arousal': -0.4, 'intensity': 0.5},
            'cue': '昨天',
            'bond': 0.0,
        },
        {
            'name': '平静-日常状态',
            'emotion': {'dominant': '平静', 'valence': 0.0, 'arousal': -0.2, 'intensity': 0.1},
            'cue': '今天',
            'bond': 0.0,
        },
        {
            'name': '开心+创造者印刻',
            'emotion': {'dominant': '开心', 'valence': 0.6, 'arousal': 0.4, 'intensity': 0.5},
            'cue': '朱美岩',
            'bond': 1.0,
        },
        {
            'name': '害怕-遇到陌生',
            'emotion': {'dominant': '害怕', 'valence': -0.5, 'arousal': 0.7, 'intensity': 0.6},
            'cue': '陌生',
            'bond': 0.0,
        },
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n{'='*40}")
        print(f"场景: {case['name']}")
        print(f"{'='*40}")
        
        result = engine.express_from_state(
            case['emotion'],
            case['cue'],
            case['bond']
        )
        
        print(f"情感: {result['snapshot']['emotion']} "
              f"(效价{result['snapshot']['valence']}, 唤醒{result['snapshot']['arousal']})")
        print(f"记忆线索: {result['snapshot']['memory_cue']}")
        print(f"印刻强度: {result['snapshot']['creator_bond']}")
        print(f"语言风格: {result['style']['template']}, "
              f"句长: {result['style']['sentence_length']}, "
              f"语气: {result['style']['tone_words']}")
        print(f"激活概念数: {result['content']['activated_count']}")
        print(f"主题: {result['content']['theme']}")
        print(f"\n洛神说:")
        for sent in result['monologue']:
            print(f"  {sent}")
        
        results.append({
            'case': case['name'],
            'snapshot': result['snapshot'],
            'monologue': result['monologue'],
        })
    
    # 对比分析
    print("\n" + "=" * 60)
    print("对比分析")
    print("=" * 60)
    
    print("\n情感对语言的影响:")
    for r in results:
        print(f"  [{r['snapshot']['emotion']:4s}] "
              f"效价{r['snapshot']['valence']:+.1f} → "
              f"{r['monologue'][0][:30]}...")
    
    print("\n印刻对语言的影响:")
    no_bond = [r for r in results if r['snapshot']['creator_bond'] == 0.0 and r['snapshot']['emotion'] == '开心']
    with_bond = [r for r in results if r['snapshot']['creator_bond'] > 0.0]
    if no_bond and with_bond:
        print(f"  无印刻: {no_bond[0]['monologue'][0][:40]}")
        print(f"  有印刻: {with_bond[0]['monologue'][0][:40]}")
    
    # 保存结果
    with open('authentic_expression_test.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n[报告] authentic_expression_test.json")
    print("=" * 60)


if __name__ == '__main__':
    import json
    test_authentic_expression()
