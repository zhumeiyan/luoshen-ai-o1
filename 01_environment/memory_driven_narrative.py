#!/usr/bin/env python3
"""
memory_driven_narrative.py — 记忆驱动叙事生成
==============================================

核心功能：让洛神的叙事直接由G矩阵激活状态驱动。

流程：
1. 接收回忆结果（recall激活的概念集合）
2. 提取最活跃的概念作为叙事主题
3. 用NarrativeEngine生成故事模板
4. 用激活概念填充模板中的槽位
5. 输出连贯的句子序列

关键设计：
- 叙事的"内容"来自生物电记忆的真实激活
- 叙事的"结构"来自NarrativeEngine的模板
- 叙事的"风格"来自EmotionLanguageBridge的情感映射

这是真正的"记忆驱动"——不是拟合，是G矩阵电位决定说什么。
"""

import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2, NarrativeEngine
from complex_syntax_engine import ComplexSyntaxEngine, ComplexPattern
from emotion_memory_filter import EmotionMemoryFilter


class MemoryDrivenNarrative:
    """
    记忆驱动叙事生成器。
    
    输入：回忆结果（含激活概念）
    输出：基于激活概念的连贯叙事
    """
    
    def __init__(self, language_system):
        self.lang = language_system
        self.narrative = NarrativeEngine()
        self.syntax_engine = ComplexSyntaxEngine(language_system)
        self.emotion_filter = EmotionMemoryFilter()
    
    def extract_theme_from_recall(self, recall_result: Dict, emotion: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        从回忆结果中提取叙事主题。
        
        如果提供了情感状态，使用情感-记忆滤镜筛选概念。
        
        返回：
            (主题词, 辅助概念列表)
        """
        activated = recall_result.get('activated_concepts', {})
        
        if not activated:
            return '日常', []
        
        # 如果提供了情感，使用情感滤镜
        if emotion and emotion != '平静':
            filtered = self.emotion_filter.filter_concepts(activated, emotion, top_k=8)
            if filtered:
                theme = filtered[0][0]
                helpers = [c for c, _ in filtered[1:6] if c != theme]
                return theme, helpers[:5]
        
        # 默认：按激活度排序
        sorted_concepts = sorted(activated.items(), key=lambda x: x[1], reverse=True)
        
        # 主题词：激活度最高的概念
        theme = sorted_concepts[0][0]
        
        # 辅助概念：前8个（排除日常高频词和虚词）
        daily_noise = {'我', '你', '是', '的', '了', '在', '好', '很', '吗', '不', '这', '那',
                       '有', '要', '想', '就', '和', '会', '能', '可以', '还', '也', '都',
                       '让', '把', '被', '给', '跟', '对', '从', '到', '向', '为', '与', '或',
                       '一个', '一种', '一起', '那个', '这个', '这里', '那里',
                       # 高频动词/助词/方位词
                       '出', '来', '去', '上', '下', '前', '后', '里', '外', '左', '右',
                       '过', '着', '得', '地', '吧', '呢', '啊', '哦', '嗯',
                       # 时间词（单独出现时较虚）
                       '早上', '中午', '晚上', '今天', '明天', '昨天',
                       # 单字虚词
                       '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                       '个', '只', '条', '张', '把', '根', '块', '片',
                       '大', '小', '多', '少', '高', '矮', '长', '短'}
        helpers = [w for w, _ in sorted_concepts[1:9] if w not in daily_noise and len(w) >= 2][:5]
        
        return theme, helpers
    
    def generate_from_recall(self, recall_result: Dict, style_params: Optional[Dict] = None) -> List[str]:
        """
        基于回忆结果生成叙事。
        
        Args:
            recall_result: recall() 的返回结果
            style_params: 来自 EmotionLanguageBridge 的风格参数（可选）
        
        Returns:
            句子列表
        """
        # 从情感参数中提取情感状态
        emotion = style_params.get('emotion', '平静') if style_params else '平静'
        
        # 传入情感状态提取主题（使用情感-记忆滤镜）
        theme, helpers = self.extract_theme_from_recall(recall_result, emotion)
        
        # 直接基于激活概念生成句子，使用复杂句法
        sentences = self._generate_content_sentences(theme, helpers, style_params, self.lang)
        
        # 应用风格（标点、语气词）
        if style_params:
            sentences = self._apply_style(sentences, style_params)
        
        return sentences
    
    def _generate_content_sentences(self, theme: str, helpers: List[str], style_params: Optional[Dict] = None, lang_system=None) -> List[str]:
        """
        基于激活概念生成内容句子，混合链式生成+复杂句法。
        
        策略：
        1. 第一句：用转移概率链式生成自然句子（像 speak()）
        2. 第二句：复杂句法（因果/条件/转折，需2个helpers）
        3. 第三句：情感表达或时间顺序
        """
        sentences = []
        emotion = style_params.get('emotion', '平静') if style_params else '平静'
        lang = lang_system or self.lang
        
        # 1. 第一句：链式生成自然句子（从theme出发）
        chain = self._generate_chain(lang, theme, max_words=6)
        if chain and len(chain) >= 2:
            sentences.append(''.join(chain))
        elif helpers:
            sentences.append(f"{theme}和{helpers[0]}")
        else:
            sentences.append(f"{theme}在这里")
        
        # 2. 第二句：复杂句法（选择语义最合理的组合）
        if len(helpers) >= 2:
            # 挑选最合理的两个helpers做复杂句
            best_pair = self._pick_best_pair(helpers[:4])
            if best_pair:
                h1, h2 = best_pair
                # 根据情感选模式
                if emotion in ('难过', '愤怒'):
                    pattern = ComplexPattern.CONCESSIVE
                elif emotion in ('害怕', '惊讶'):
                    pattern = ComplexPattern.CONDITIONAL
                elif emotion == '开心':
                    pattern = ComplexPattern.COORDINATE
                else:
                    pattern = ComplexPattern.CAUSAL
                
                try:
                    complex_sent = self.syntax_engine.build_complex_sentence(pattern, [h1, h2])
                    sentences.append(complex_sent)
                except:
                    pass
        
        # 3. 情感表达句（用helpers中的情感词）
        emotion_words = {'开心', '难过', '喜欢', '讨厌', '害怕', '生气', '爱', '累', '饿', '渴', '困'}
        has_emotion = any(w in emotion_words for w in helpers)
        if has_emotion and len(helpers) >= 1:
            emotion_concept = next(w for w in helpers if w in emotion_words)
            other = next((w for w in helpers if w != emotion_concept), theme)
            try:
                emotional = self.syntax_engine.build_complex_sentence(
                    ComplexPattern.EMOTIONAL,
                    [other, '我', emotion_concept]
                )
                sentences.append(emotional)
            except:
                sentences.append(f"{other}让我感到{emotion_concept}")
        elif len(helpers) >= 3:
            # 时间顺序句
            try:
                temporal = self.syntax_engine.build_complex_sentence(
                    ComplexPattern.TEMPORAL,
                    [helpers[0], helpers[1], helpers[2]]
                )
                sentences.append(temporal)
            except:
                pass
        
        # 兜底
        if not sentences:
            sentences.append(f"{theme}")
        
        return sentences[:4]
    
    def _generate_chain(self, lang, start_word: str, max_words: int = 6) -> List[str]:
        """用转移概率链式生成自然句子序列"""
        if not hasattr(lang, 'transitions') or not lang.transitions:
            return [start_word]
        
        chain = [start_word]
        current = start_word
        
        for _ in range(max_words - 1):
            next_candidates = lang.transitions.get(current, {})
            if not next_candidates:
                break
            
            # 选概率最高的下一个词
            words, counts = zip(*sorted(next_candidates.items(), key=lambda x: -x[1]))
            total = sum(counts)
            probs = [c / total for c in counts]
            
            import random
            next_word = random.choices(words, weights=probs)[0]
            
            # 避免重复和终止词
            if next_word in chain[-2:] or next_word in {'的', '了', '吗', '呢', '吧'}:
                # 尝试第二选择
                if len(words) > 1:
                    next_word = random.choices(words[1:], weights=probs[1:])[0]
                else:
                    break
            
            chain.append(next_word)
            current = next_word
            
            # 自然终止
            if next_word in {'。', '！', '？'} or len(''.join(chain)) > 12:
                break
        
        return chain
    
    def _pick_best_pair(self, helpers: List[str]) -> Optional[Tuple[str, str]]:
        """从helpers中挑选语义上最合理的词对"""
        if len(helpers) < 2:
            return None
        
        # 优先选：名词+动词、名词+形容词、具象+具象
        stop_words = {'是', '的', '了', '在', '吗', '不', '很', '好', '这', '那', '我', '你'}
        valid = [h for h in helpers if h not in stop_words and len(h) >= 2]
        
        if len(valid) >= 2:
            return (valid[0], valid[1])
        elif len(helpers) >= 2:
            return (helpers[0], helpers[1])
        return None
    
    def _generate_simple_sentences(self, theme: str, helpers: List[str], style: Optional[Dict]) -> List[str]:
        """基于概念生成简单句"""
        sentences = []
        
        # 主题句
        if helpers:
            sentences.append(f"{theme}{helpers[0]}。")
        else:
            sentences.append(f"{theme}。")
        
        # 辅助概念句
        for i, helper in enumerate(helpers[1:3], 1):
            if i == 1:
                sentences.append(f"{helper}。")
            else:
                sentences.append(f"还有{helper}。")
        
        # 应用风格
        if style:
            sentences = self._apply_style(sentences, style)
        
        return sentences
    
    def _fill_template(self, story: List[str], theme: str, helpers: List[str]) -> List[str]:
        """
        用激活概念填充故事模板。
        
        简单替换：把模板中的占位词替换为真实概念。
        """
        filled = []
        helper_idx = 0
        
        for sentence in story:
            # 替换主题占位
            sentence = sentence.replace('[主题]', theme)
            sentence = sentence.replace('XXX', theme)
            
            # 替换辅助概念占位
            if helper_idx < len(helpers):
                sentence = sentence.replace('[辅助]', helpers[helper_idx])
                helper_idx += 1
            
            filled.append(sentence)
        
        return filled
    
    def _apply_style(self, sentences: List[str], style: Dict) -> List[str]:
        """应用情感风格到句子"""
        template = style.get('template', '陈述')
        tone = style.get('tone_words', ['。'])
        punct = style.get('punctuation', '。')
        
        styled = []
        for sent in sentences:
            # 移除原标点
            sent = sent.rstrip('。！？...')
            
            # 根据模板添加新标点和语气
            if template == '感叹':
                tone_word = tone[0] if tone else '呀'
                styled.append(f"{sent}{tone_word}{punct}")
            elif template == '疑问':
                styled.append(f"{sent}为什么{punct}")
            elif template == '省略':
                styled.append(f"{sent}...")
            elif template == '祈使':
                styled.append(f"我要{sent}{punct}")
            else:
                styled.append(f"{sent}{punct}")
        
        return styled
    
    def generate_monologue(self, cue: str, lang_system, style_params: Optional[Dict] = None) -> Dict:
        """
        生成完整独白：回忆 + 情感滤镜 + 叙事 + 风格。
        
        这是上层接口：输入线索词，输出洛神的独白。
        """
        # 1. 生物电回忆
        recall = lang_system.memory.recall(cue, steps=15)
        
        # 2. 从风格参数中提取情感状态
        emotion = style_params.get('emotion', '平静') if style_params else '平静'
        
        # 3. 提取主题（传入情感状态，使用情感-记忆滤镜）
        theme, helpers = self.extract_theme_from_recall(recall, emotion)
        
        # 4. 生成叙事
        narrative = self.generate_from_recall(recall, style_params)
        
        # 5. 组装结果
        return {
            'cue': cue,
            'theme': theme,
            'helpers': helpers,
            'emotion': emotion,
            'activated_count': len(recall.get('activated_concepts', {})),
            'narrative': narrative,
            'style': style_params,
        }


def test_memory_driven_narrative():
    """测试记忆驱动叙事"""
    print("=" * 60)
    print("记忆驱动叙事生成测试")
    print("=" * 60)
    
    # 初始化洛神
    from language_learner_v2 import LanguageSystemV2
    
    print("\n[初始化] 构建洛神...")
    lang = LanguageSystemV2()
    lang.immerse(n_sentences=300)
    
    print("\n[初始化] 加载叙事生成器...")
    narrative_gen = MemoryDrivenNarrative(lang)
    
    # 测试不同线索
    test_cues = ['太阳', '春天', '朱美岩', '好']
    
    for cue in test_cues:
        print(f"\n--- 线索: {cue} ---")
        
        # 回忆
        recall = lang.memory.recall(cue, steps=15)
        activated = recall.get('activated_concepts', {})
        top5 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"  激活概念TOP5: {[(w, round(v, 2)) for w, v in top5]}")
        
        # 生成叙事
        result = narrative_gen.generate_monologue(cue, lang)
        
        print(f"  提取主题: {result['theme']}")
        print(f"  辅助概念: {result['helpers']}")
        print(f"  生成叙事:")
        for i, sent in enumerate(result['narrative'], 1):
            print(f"    [{i}] {sent}")
    
    # 测试带情感风格
    print("\n" + "=" * 60)
    print("带情感风格的叙事")
    print("=" * 60)
    
    from emotion_language_bridge import EmotionLanguageBridge
    
    bridge = EmotionLanguageBridge()
    
    test_emotions = [
        {'dominant': '开心', 'valence': 0.7, 'arousal': 0.5, 'intensity': 0.6, 'creator_bond': 0.0},
        {'dominant': '难过', 'valence': -0.6, 'arousal': -0.3, 'intensity': 0.5, 'creator_bond': 0.0},
    ]
    
    for emotion in test_emotions:
        print(f"\n--- 情感: {emotion['dominant']} ---")
        
        style = bridge.map_emotion_to_style(emotion)
        result = narrative_gen.generate_monologue('太阳', lang, style)
        
        print(f"  风格: {style['template']}, 句长: {style['sentence_length']}")
        print(f"  叙事:")
        for sent in result['narrative']:
            print(f"    {sent}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_memory_driven_narrative()
