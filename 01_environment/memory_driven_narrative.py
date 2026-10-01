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
import random
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
        
        # 强力虚词过滤表（统一的过滤标准）
        stop_words = {'我', '你', '他', '她', '它', '我们', '你们', '他们',
                      '是', '的', '了', '在', '吗', '不', '很', '也', '都', '就',
                      '和', '有', '要', '想', '会', '能', '还', '让', '把', '被',
                      '给', '跟', '对', '从', '到', '向', '为', '与', '或',
                      '又', '再', '没', '别', '太', '真', '比较', '非常', '特别',
                      '有点', '那么', '这么', '什么', '怎么', '为什么',
                      '因为', '所以', '如果', '虽然', '但是', '然后', '最后',
                      '而且', '要么', '不但', '不仅',
                      '呀', '啊', '呢', '吧', '哦', '嗯', '哇', '哈',
                      '出', '来', '去', '上', '下', '前', '后', '里', '外',
                      '个', '只', '条', '张', '把', '根', '块', '片',
                      '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                      '这', '那', '这里', '那里', '这个', '那个', '这些', '那些',
                      '今天', '明天', '昨天', '早上', '中午', '晚上',
                      '可以', '可能', '也许', '大概', '一定', '必须'}
        
        # 统一过滤函数
        def is_content(word):
            return word not in stop_words and len(word) >= 2
        
        # 如果提供了情感，使用情感滤镜
        if emotion and emotion != '平静':
            filtered = self.emotion_filter.filter_concepts(activated, emotion, top_k=8)
            # 再次过滤虚词
            filtered = [(c, s) for c, s in filtered if is_content(c)]
            if filtered:
                theme = filtered[0][0]
                helpers = [c for c, _ in filtered[1:6] if c != theme and is_content(c)]
                return theme, helpers[:5]
        
        # 默认：按激活度排序，但强力过滤虚词
        # 完整的虚词表
        stop_words = {'我', '你', '他', '她', '它', '我们', '你们', '他们',
                      '是', '的', '了', '在', '吗', '不', '很', '也', '都', '就',
                      '和', '有', '要', '想', '会', '能', '还', '让', '把', '被',
                      '给', '跟', '对', '从', '到', '向', '为', '与', '或',
                      '又', '再', '没', '别', '太', '真', '比较', '非常', '特别',
                      '有点', '那么', '这么', '什么', '怎么', '为什么',
                      '因为', '所以', '如果', '虽然', '但是', '然后', '最后',
                      '而且', '要么', '不但', '不仅',
                      '呀', '啊', '呢', '吧', '哦', '嗯', '哇', '哈',
                      '出', '来', '去', '上', '下', '前', '后', '里', '外',
                      '个', '只', '条', '张', '把', '根', '块', '片',
                      '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                      '这', '那', '这里', '那里', '这个', '那个', '这些', '那些',
                      '今天', '明天', '昨天', '早上', '中午', '晚上',
                      '可以', '可能', '也许', '大概', '一定', '必须'}
        
        # 过滤后的排序
        filtered = [(w, s) for w, s in sorted(activated.items(), key=lambda x: x[1], reverse=True)
                    if w not in stop_words and len(w) >= 2]
        
        if not filtered:
            # 如果实在没有内容词，回退到原始线索
            return '日常', []
        
        # 主题词：过滤后激活度最高的内容词
        theme = filtered[0][0]
        
        # 辅助概念：过滤后的前5个
        helpers = [w for w, _ in filtered[1:6]]
        
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
        基于激活概念生成内容句子。
        
        策略：不用不稳定的链式生成，改用模板填充+复杂句法，
        确保每句都有明确语义。
        """
        sentences = []
        emotion = style_params.get('emotion', '平静') if style_params else '平静'
        
        # 过滤helpers：只保留有效内容词（>=2字，非虚词）
        stop_words = {'是', '的', '了', '在', '吗', '不', '很', '也', '都', '就',
                      '和', '有', '要', '想', '会', '能', '还', '让', '把', '被',
                      '给', '跟', '对', '从', '到', '向', '为', '与', '或',
                      '又', '再', '没', '别', '太', '真', '比较', '非常', '特别',
                      '有点', '那么', '这么', '什么', '怎么', '为什么',
                      '因为', '所以', '如果', '虽然', '但是', '然后', '最后',
                      '我', '你', '他', '她', '它', '我们', '你们', '他们',
                      '呀', '啊', '呢', '吧', '哦', '嗯',
                      '出', '来', '去', '上', '下', '前', '后', '里', '外',
                      '个', '只', '条', '张', '把', '根', '块', '片',
                      '可以', '可能', '也许', '大概', '一定', '必须'}
        
        valid_helpers = [h for h in helpers if h not in stop_words and len(h) >= 2]
        
        # 1. 第一句：主题 + 简单描述（从helpers中找一个搭配）
        if valid_helpers:
            h1 = valid_helpers[0]
            # 情感适配的谓词
            predicates = {
                '开心': ['很好', '让人开心', '很漂亮', '很有趣'],
                '难过': ['让人难过', '有点伤心', '不太好', '让人想哭'],
                '害怕': ['有点可怕', '让人害怕', '很吓人', '不安全'],
                '愤怒': ['让人生气', '很讨厌', '不好', '真烦人'],
                '平静': ['在这里', '很安静', '很好', '很美'],
                '好奇': ['是什么', '很有趣', '很好奇', '想知道'],
            }
            preds = predicates.get(emotion, ['很好', '在这里', '很美'])
            pred = random.choice(preds)
            sentences.append(f"{theme}{pred}")
        else:
            sentences.append(f"{theme}在这里")
        
        # 2. 第二句：复杂句法（如果helpers >= 2）
        if len(valid_helpers) >= 2:
            h1, h2 = valid_helpers[0], valid_helpers[1]
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
        
        # 3. 第三句：情感表达或简单陈述
        emotion_words = {'开心', '难过', '喜欢', '讨厌', '害怕', '生气', '爱', '累', '饿', '渴', '困'}
        has_emotion = any(w in valid_helpers for w in emotion_words)
        if has_emotion and len(valid_helpers) >= 1:
            emotion_concept = next(w for w in valid_helpers if w in emotion_words)
            other = next((w for w in valid_helpers if w != emotion_concept), theme)
            try:
                emotional = self.syntax_engine.build_complex_sentence(
                    ComplexPattern.EMOTIONAL,
                    [other, '我', emotion_concept]
                )
                sentences.append(emotional)
            except:
                sentences.append(f"{other}让我感到{emotion_concept}")
        elif len(valid_helpers) >= 1:
            # 简单情感陈述
            if emotion == '开心':
                sentences.append(f"我喜欢{valid_helpers[0]}")
            elif emotion == '难过':
                sentences.append(f"{valid_helpers[0]}让我难过")
            elif emotion == '害怕':
                sentences.append(f"{valid_helpers[0]}有点可怕")
            else:
                sentences.append(f"{valid_helpers[0]}和{theme}一起")
        
        # 兜底
        if not sentences:
            sentences.append(f"{theme}")
        
        return sentences[:4]
    
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
