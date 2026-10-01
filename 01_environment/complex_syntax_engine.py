#!/usr/bin/env python3
"""
complex_syntax_engine.py — 复杂句法引擎
=========================================

目标：在现有5种句法模式基础上，增加复合句、条件句、因果句等复杂结构。

现有句法模式（language_learner_v2）：
- SVO: 主谓宾（"我吃苹果"）
- 疑问: "吗"结尾
- 复合: 多个SVO
- 否定: "不"
- 祈使: 命令式

新增复杂句法：
- 因果句: "因为...所以..."
- 条件句: "如果...就..."
- 转折句: "虽然...但是..."
- 并列句: "一边...一边..."
- 递进句: "不仅...而且..."
- 假设句: "要是...的话..."
- 选择句: "是...还是..."
- 时间顺序: "先...再...最后..."

关键设计：
- 复杂句法不是简单拼接，而是基于概念间的语义关系
- 因果句需要SemanticEngine中的因果规则
- 条件句需要假设-结果的概念关联
- 所有复杂句法都基于G矩阵中的共现连接构建
"""

import sys
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto

sys.path.insert(0, '01_environment')


class ComplexPattern(Enum):
    """复杂句法模式"""
    CAUSAL = "因果"           # 因为A，所以B
    CONDITIONAL = "条件"      # 如果A，就B
    CONCESSIVE = "转折"       # 虽然A，但是B
    COORDINATE = "并列"       # A，而且B
    PROGRESSIVE = "递进"      # 不仅A，而且B
    HYPOTHETICAL = "假设"     # 要是A的话，B
    ALTERNATIVE = "选择"      # 是A，还是B
    TEMPORAL = "时间顺序"     # 先A，再B，最后C
    DESCRIPTIVE = "描写"      # A像B一样C
    EMOTIONAL = "情感表达"    # A让B感到C


class ComplexSyntaxEngine:
    """
    复杂句法引擎。
    
    基于G矩阵中的概念关联，构建复杂句法结构。
    """
    
    def __init__(self, language_system=None):
        self.lang = language_system
        
        # 句法模板库
        self.templates = self._build_templates()
        
        # 连接词库
        self.connectors = self._build_connectors()
    
    def _build_templates(self) -> Dict[ComplexPattern, List[str]]:
        """构建复杂句法模板"""
        return {
            ComplexPattern.CAUSAL: [
                "因为{clause1}，所以{clause2}",
                "{clause1}，所以{clause2}",
                "由于{clause1}，{clause2}",
            ],
            ComplexPattern.CONDITIONAL: [
                "如果{clause1}，就{clause2}",
                "要是{clause1}，就{clause2}",
                "只要{clause1}，就{clause2}",
            ],
            ComplexPattern.CONCESSIVE: [
                "虽然{clause1}，但是{clause2}",
                "尽管{clause1}，还是{clause2}",
                "{clause1}，不过{clause2}",
            ],
            ComplexPattern.COORDINATE: [
                "{clause1}，而且{clause2}",
                "{clause1}，也{clause2}",
                "{clause1}，同时{clause2}",
            ],
            ComplexPattern.PROGRESSIVE: [
                "不仅{clause1}，而且{clause2}",
                "不但{clause1}，还{clause2}",
                "除了{clause1}，还{clause2}",
            ],
            ComplexPattern.HYPOTHETICAL: [
                "要是{clause1}的话，{clause2}",
                "假如{clause1}，{clause2}",
                "万一{clause1}，{clause2}",
            ],
            ComplexPattern.ALTERNATIVE: [
                "是{clause1}，还是{clause2}",
                "要么{clause1}，要么{clause2}",
                "{clause1}好呢，还是{clause2}好呢",
            ],
            ComplexPattern.TEMPORAL: [
                "先{clause1}，再{clause2}，最后{clause3}",
                "{clause1}之后，{clause2}",
                "在{clause1}之前，{clause2}",
            ],
            ComplexPattern.DESCRIPTIVE: [
                "{subject}像{object}一样{action}",
                "{subject}仿佛{object}般{action}",
            ],
            ComplexPattern.EMOTIONAL: [
                "{subject}让{object}感到{action}",
                "{object}使{subject}{action}",
                "看到{subject}，{object}{action}",
            ],
        }
    
    def _build_connectors(self) -> Dict[str, List[str]]:
        """构建连接词库"""
        return {
            'cause': ['因为', '由于', '所以', '因此', '导致'],
            'condition': ['如果', '要是', '只要', '除非'],
            'contrast': ['虽然', '但是', '不过', '然而', '尽管'],
            'coordinate': ['而且', '也', '同时', '并且'],
            'progressive': ['不仅', '而且', '还', '甚至'],
            'hypothetical': ['假如', '万一', '要是'],
            'alternative': ['或者', '还是', '要么'],
            'temporal': ['先', '再', '最后', '之后', '之前'],
        }
    
    def detect_pattern(self, concept1: str, concept2: str, lang_system=None) -> Optional[ComplexPattern]:
        """
        检测两个概念之间最适合的复杂句法模式。
        
        基于G矩阵中的共现关系和语义引擎。
        """
        if not lang_system:
            # 默认返回因果模式
            return ComplexPattern.CAUSAL
        
        # 检查语义引擎中的因果规则
        if hasattr(lang_system, 'semantic') and lang_system.semantic:
            # 检查是否有因果规则
            causes = lang_system.semantic.infer_cause(concept2)
            if concept1 in causes:
                return ComplexPattern.CAUSAL
        
        # 检查共现频率判断关联类型
        if hasattr(lang_system, 'transitions'):
            # 双向高共现 → 并列
            c1_to_c2 = lang_system.transitions.get(concept1, {}).get(concept2, 0)
            c2_to_c1 = lang_system.transitions.get(concept2, {}).get(concept1, 0)
            
            if c1_to_c2 > 5 and c2_to_c1 > 5:
                return ComplexPattern.COORDINATE
            elif c1_to_c2 > 5:
                return ComplexPattern.CAUSAL
        
        # 根据概念特性判断
        emotion_words = {'开心', '难过', '喜欢', '讨厌', '害怕', '生气'}
        if concept1 in emotion_words or concept2 in emotion_words:
            return ComplexPattern.EMOTIONAL
        
        action_words = {'吃', '走', '看', '听', '跑', '跳', '睡', '玩'}
        if concept1 in action_words and concept2 in action_words:
            return ComplexPattern.TEMPORAL
        
        # 默认因果
        return ComplexPattern.CAUSAL
    
    def build_complex_sentence(self, pattern: ComplexPattern, concepts: List[str], lang_system=None) -> str:
        """
        构建复杂句子。
        
        Args:
            pattern: 复杂句法模式
            concepts: 概念列表（2-3个）
            lang_system: 语言系统（可选）
        """
        templates = self.templates.get(pattern, [])
        if not templates:
            return "..."
        
        template = templates[0]  # 使用第一个模板
        
        if pattern == ComplexPattern.TEMPORAL and len(concepts) >= 3:
            # 时间顺序需要3个概念
            return template.format(
                clause1=concepts[0],
                clause2=concepts[1],
                clause3=concepts[2]
            )
        elif pattern in [ComplexPattern.DESCRIPTIVE, ComplexPattern.EMOTIONAL] and len(concepts) >= 3:
            return template.format(
                subject=concepts[0],
                object=concepts[1],
                action=concepts[2]
            )
        elif len(concepts) >= 2:
            # 大多数模式需要2个概念
            return template.format(
                clause1=concepts[0],
                clause2=concepts[1]
            )
        else:
            # 只有一个概念，简单陈述
            return f"{concepts[0]}。"
    
    def generate_from_recall(self, recall_result: Dict, emotion_style: Optional[Dict] = None) -> List[str]:
        """
        基于回忆结果生成复杂句子。
        
        从激活概念中提取主题和辅助概念，构建复杂句法。
        """
        activated = recall_result.get('activated_concepts', {})
        if not activated:
            return ["..."]
        
        # 获取前几个最活跃的概念
        sorted_concepts = sorted(activated.items(), key=lambda x: x[1], reverse=True)
        
        # 过滤日常高频词
        daily_noise = {'我', '你', '是', '的', '了', '在', '好', '很', '吗', '不', '这', '那'}
        meaningful = [w for w, _ in sorted_concepts if w not in daily_noise][:6]
        
        if len(meaningful) < 2:
            # 概念不足，返回简单句
            if meaningful:
                return [f"{meaningful[0]}。"]
            return ["..."]
        
        sentences = []
        
        # 生成不同类型的复杂句
        # 1. 因果句（前两个概念）
        if len(meaningful) >= 2:
            pattern = self.detect_pattern(meaningful[0], meaningful[1])
            sentence = self.build_complex_sentence(pattern, meaningful[:2])
            sentences.append(sentence)
        
        # 2. 并列/递进句（第2-3个概念）
        if len(meaningful) >= 3:
            sentence = self.build_complex_sentence(
                ComplexPattern.COORDINATE,
                meaningful[1:3]
            )
            sentences.append(sentence)
        
        # 3. 时间顺序（如果有3个以上概念）
        if len(meaningful) >= 3:
            sentence = self.build_complex_sentence(
                ComplexPattern.TEMPORAL,
                meaningful[:3]
            )
            sentences.append(sentence)
        
        # 4. 情感表达（如果有情感词）
        emotion_words = {'开心', '难过', '喜欢', '讨厌', '害怕', '生气', '爱'}
        has_emotion = any(w in emotion_words for w in meaningful)
        if has_emotion and len(meaningful) >= 2:
            emotion_concept = next(w for w in meaningful if w in emotion_words)
            other = next(w for w in meaningful if w != emotion_concept)
            sentence = self.build_complex_sentence(
                ComplexPattern.EMOTIONAL,
                [other, '我', emotion_concept]
            )
            sentences.append(sentence)
        
        return sentences[:3]  # 最多返回3句
    
    def generate_monologue(self, theme: str, helpers: List[str], emotion: str = '平静') -> List[str]:
        """
        生成连贯独白。
        
        根据主题、辅助概念和情感，构建多句独白。
        """
        sentences = []
        
        # 第一句：引入主题
        if emotion == '开心':
            sentences.append(f"{theme}真好呀！")
        elif emotion == '难过':
            sentences.append(f"{theme}...让人难过...")
        elif emotion == '愤怒':
            sentences.append(f"{theme}！气死我了！")
        elif emotion == '害怕':
            sentences.append(f"{theme}...好可怕...")
        else:
            sentences.append(f"{theme}。")
        
        # 第二句：复杂句法（因果/并列）
        if len(helpers) >= 2:
            pattern = ComplexPattern.CAUSAL if emotion == '难过' else ComplexPattern.COORDINATE
            sentence = self.build_complex_sentence(pattern, [theme, helpers[0]])
            sentences.append(sentence)
        
        # 第三句：情感表达
        if emotion == '开心' and helpers:
            sentences.append(f"看到{helpers[0]}，我很开心！")
        elif emotion == '难过' and helpers:
            sentences.append(f"虽然{helpers[0]}，但是{theme}还在。")
        elif helpers:
            sentences.append(f"{theme}和{helpers[0]}一起。")
        
        return sentences


def test_complex_syntax():
    """测试复杂句法引擎"""
    print("=" * 60)
    print("复杂句法引擎测试")
    print("=" * 60)
    
    engine = ComplexSyntaxEngine()
    
    # 测试1：各种复杂句法模式
    print("\n--- 测试1：复杂句法模板 ---")
    
    test_cases = [
        (ComplexPattern.CAUSAL, ['下雨', '地湿'], "因果"),
        (ComplexPattern.CONDITIONAL, ['努力', '成功'], "条件"),
        (ComplexPattern.CONCESSIVE, ['累', '坚持'], "转折"),
        (ComplexPattern.COORDINATE, ['聪明', '勤奋'], "并列"),
        (ComplexPattern.PROGRESSIVE, ['学习', '实践'], "递进"),
        (ComplexPattern.HYPOTHETICAL, ['有钱', '旅行'], "假设"),
        (ComplexPattern.ALTERNATIVE, ['咖啡', '茶'], "选择"),
        (ComplexPattern.TEMPORAL, ['起床', '刷牙', '吃饭'], "时间"),
        (ComplexPattern.DESCRIPTIVE, ['月亮', '银盘', '挂天上'], "描写"),
        (ComplexPattern.EMOTIONAL, ['花', '我', '开心'], "情感"),
    ]
    
    for pattern, concepts, name in test_cases:
        sentence = engine.build_complex_sentence(pattern, concepts)
        print(f"  [{name:4s}] {sentence}")
    
    # 测试2：基于回忆生成
    print("\n--- 测试2：基于回忆生成 ---")
    
    # 模拟回忆结果
    mock_recall = {
        'activated_concepts': {
            '太阳': 2.5, '温暖': 1.8, '开心': 1.5, '花': 1.2, '春天': 1.0,
            '我': 0.8, '是': 0.5, '的': 0.3,
        }
    }
    
    sentences = engine.generate_from_recall(mock_recall)
    print("  回忆主题: 太阳/温暖/开心/花/春天")
    print("  生成句子:")
    for i, sent in enumerate(sentences, 1):
        print(f"    [{i}] {sent}")
    
    # 测试3：不同情感的独白
    print("\n--- 测试3：不同情感的独白 ---")
    
    for emotion in ['开心', '难过', '平静']:
        sentences = engine.generate_monologue('太阳', ['温暖', '花'], emotion)
        print(f"\n  情感: {emotion}")
        for sent in sentences:
            print(f"    {sent}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_complex_syntax()
