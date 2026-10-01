#!/usr/bin/env python3
"""
context_tracker.py — 上下文追踪引擎
==============================

在 DialogueEngine 基础上增强多轮对话记忆能力：

核心功能：
1. 多轮记忆缓冲：维护最近 N 轮对话的完整上下文
2. 话题延续：检测当前输入是否延续前文话题
3. 指代消解：将"它/这/那"等指代词映射到前文实体
4. 情感连贯：让情感状态在对话中自然延续/转换
5. 上下文感知的回应生成：基于前文调整当前回应

设计原则：
- 不是简单存储原文，而是提取关键概念和关系
- 上下文影响的是"说什么"和"怎么说"
- 与生物电记忆系统协同工作
"""

from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import sys

sys.path.insert(0, '01_environment')


class ContextTracker:
    """
    上下文追踪引擎。
    
    维护多轮对话的语义上下文，让洛神的回应连贯自然。
    """
    
    def __init__(self, max_history: int = 8):
        # 对话历史（完整轮次）
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        
        # 话题栈：支持嵌套话题（主话题+子话题）
        self.topic_stack = []
        self.topic_depth = 3  # 最多嵌套3层
        
        # 实体记忆：前文提到的关键实体（名词/概念）
        self.entity_buffer = deque(maxlen=20)  # 最近20个实体
        
        # 情感轨迹：对话中的情感变化
        self.emotion_trace = deque(maxlen=max_history)
        
        # 未回应的问题/请求（需要后续跟进）
        self.pending_questions = []
        
        # 连贯性状态
        self.coherence_state = {
            'last_cue': None,      # 上一次的回忆线索
            'last_emotion': None,  # 上一次的情感状态
            'repetition_count': 0, # 连续重复同一话题的次数
        }
    
    def add_turn(self, speaker: str, text: str, 
                 concepts: Optional[List[str]] = None,
                 emotion: Optional[str] = None):
        """
        添加一轮对话到上下文。
        
        Args:
            speaker: 'user' 或 'luoshen'
            text: 对话内容
            concepts: 提取的关键概念列表
            emotion: 本轮的情感状态
        """
        turn = {
            'speaker': speaker,
            'text': text,
            'concepts': concepts or [],
            'emotion': emotion,
            'turn_id': len(self.history),
        }
        
        self.history.append(turn)
        
        # 过滤实体：只保留看起来像实体的词（>=2字，非虚词，非纯情感词）
        STOP_WORDS = {'的', '了', '是', '吗', '不', '很', '也', '都', '就', '和',
                      '在', '有', '要', '想', '出', '来', '去', '上', '下', '这',
                      '那', '它', '他', '她', '你', '我', '们', '个', '只', '条',
                      '张', '把', '根', '块', '片', '一', '二', '三', '四', '五',
                      '六', '七', '八', '九', '十', '过', '着', '得', '地', '吧',
                      '呢', '啊', '哦', '嗯', '还', '能', '会', '让', '把', '被',
                      '给', '跟', '对', '从', '到', '向', '为', '与', '或', '点',
                      '因为', '所以', '如果', '虽然', '但是', '然后', '最后',
                      '温', '暖', '快', '乐', '高', '兴', '早', '安', '再', '见',
                      '大', '小', '多', '少', '长', '短', '胖', '瘦', '红', '绿',
                      '蓝', '黄', '白', '黑', '金', '银', '好', '坏', '真', '假',
                      '年', '月', '日', '时', '分', '秒', '天', '地', '人',
                      # 情感词（避免作为实体）
                      '开心', '难过', '害怕', '生气', '愤怒', '惊讶', '好奇',
                      '喜欢', '讨厌', '爱', '恨', '累', '饿', '渴', '困',
                      '高兴', '快乐', '伤心', '痛苦', '舒服', '难受', '紧张',
                      '放松', '兴奋', '无聊', '有趣', '没劲', '失望', '希望',
                      '满足', '遗憾', '委屈', '嫉妒', '骄傲', '害羞', '尴尬',
                      # 状态词
                      '有点', '非常', '特别', '太', '比较', '一般', '还好',
                      '确实', '真的', '假的', '大概', '也许', '可能', '一定',
                      # 时间/方位碎片
                      '早上', '中午', '晚上', '今天', '明天', '昨天',
                      '上午', '下午', '夜里', '凌晨', '刚才', '现在',
                      '以前', '以后', '之前', '之后', '中间', '旁边',
                      '上面', '下面', '里面', '外面', '前面', '后面'}
        if concepts:
            for c in concepts:
                if len(c) >= 2 and c not in STOP_WORDS and c not in self.entity_buffer:
                    self.entity_buffer.append(c)
        
        # 更新情感轨迹
        if emotion:
            self.emotion_trace.append(emotion)
        
        # 更新连贯性状态
        self.coherence_state['last_cue'] = concepts[0] if concepts else None
        self.coherence_state['last_emotion'] = emotion
    
    def get_recent_context(self, n_turns: int = 3) -> List[Dict]:
        """获取最近 n 轮对话的上下文。"""
        return list(self.history)[-n_turns:]
    
    def get_context_cue(self) -> Optional[str]:
        """
        获取当前上下文的回忆线索。
        
        策略：
        1. 如果有未回应的问题，线索是问题中的概念
        2. 如果延续当前话题，线索是当前话题关键词
        3. 如果是新话题，线索是最近一次提到的实体
        """
        if not self.history:
            return None
        
        # 1. 检查未回应的问题
        if self.pending_questions:
            last_q = self.pending_questions[-1]
            concepts = last_q.get('concepts', [])
            if concepts:
                return concepts[0]
        
        # 2. 取最近一轮的关键概念
        recent = list(self.history)[-3:]  # 最近3轮
        all_concepts = []
        for turn in recent:
            all_concepts.extend(turn.get('concepts', []))
        
        # 去重，保持顺序
        seen = set()
        unique = []
        for c in all_concepts:
            if c not in seen and c not in {'我', '你', '的', '了', '是', '吗'}:
                seen.add(c)
                unique.append(c)
        
        return unique[0] if unique else None
    
    def is_topic_continuation(self, new_concepts: List[str]) -> bool:
        """
        判断新输入是否延续当前话题。
        
        如果有概念与最近3轮的实体重叠，视为延续。
        """
        if not self.entity_buffer or not new_concepts:
            return False
        
        recent_entities = set(self.entity_buffer)
        overlap = [c for c in new_concepts if c in recent_entities]
        
        return len(overlap) > 0
    
    def resolve_reference(self, text: str) -> str:
        """
        指代消解：将"它/这/那/他/她"替换为前文实体。
        
        简单策略：用最近提到的非说话者实体替换。
        """
        # 指代词集合
        pronouns = {'它', '这', '那', '他', '她'}
        
        # 找到前文最近的一个实体（排除"我""你"和常见虚词）
        recent_entity = None
        stop_entities = {'我', '你', '我们', '你们', '的', '了', '是', '吗', '不',
                         '很', '也', '都', '就', '和', '在', '有', '要', '想',
                         '出', '来', '去', '上', '下', '这', '那'}
        for e in reversed(self.entity_buffer):
            if e not in stop_entities and len(e) >= 2:
                recent_entity = e
                break
        
        if not recent_entity:
            return text
        
        # 替换第一个出现的指代词
        for p in pronouns:
            if p in text:
                text = text.replace(p, recent_entity, 1)
                break
        
        return text
    
    def get_emotion_context(self) -> Optional[str]:
        """获取对话中的情感上下文（主导情感）。"""
        if not self.emotion_trace:
            return None
        
        # 统计最近情感
        emotion_counts = defaultdict(int)
        for e in self.emotion_trace:
            emotion_counts[e] += 1
        
        # 返回最频繁的情感
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def detect_repetition(self) -> bool:
        """检测是否连续重复同一话题（避免单调）。"""
        if len(self.history) < 3:
            return False
        
        recent = list(self.history)[-3:]
        cues = [t.get('concepts', [None])[0] for t in recent]
        
        # 如果最近3轮线索相同，视为重复
        if cues[0] and all(c == cues[0] for c in cues):
            self.coherence_state['repetition_count'] += 1
            return True
        
        self.coherence_state['repetition_count'] = 0
        return False
    
    def suggest_diversification(self) -> Optional[str]:
        """
        当检测到重复时，建议一个新话题方向。
        
        从实体缓冲中找与当前话题相关的但未最近讨论的。
        """
        if len(self.entity_buffer) < 2:
            return None
        
        # 取倒数第2个实体（不是最新的）
        candidates = list(self.entity_buffer)[:-1]
        if candidates:
            return candidates[-1]
        return None
    
    def get_stats(self) -> Dict:
        """获取上下文统计。"""
        return {
            'history_length': len(self.history),
            'entity_count': len(self.entity_buffer),
            'emotion_trace': list(self.emotion_trace),
            'current_topic': self.topic_stack[-1] if self.topic_stack else None,
            'repetition_count': self.coherence_state['repetition_count'],
        }
    
    def clear(self):
        """清空上下文（新对话开始时）。"""
        self.history.clear()
        self.topic_stack.clear()
        self.entity_buffer.clear()
        self.emotion_trace.clear()
        self.pending_questions.clear()
        self.coherence_state = {
            'last_cue': None,
            'last_emotion': None,
            'repetition_count': 0,
        }


class ContextAwareDialogue:
    """
    上下文感知对话引擎。
    
    整合 ContextTracker + AuthenticExpressionEngine，
    实现多轮连贯对话。
    """
    
    def __init__(self, language_system, expression_engine):
        self.lang = language_system
        self.expression = expression_engine
        self.context = ContextTracker(max_history=8)
    
    def chat(self, user_input: str, user_emotion: Optional[Dict] = None) -> Dict:
        """
        上下文感知的对话入口。
        
        流程：
        1. 分词提取概念
        2. 更新上下文
        3. 判断话题延续/转换
        4. 生成情感状态（延续+用户影响）
        5. 选择回忆线索（上下文优先）
        6. 生成回应
        7. 更新上下文（洛神的回应）
        """
        # 1. 分词
        tokens = self.lang.tokenizer.tokenize(user_input)
        
        # 2. 更新用户轮次到上下文
        self.context.add_turn(
            speaker='user',
            text=user_input,
            concepts=tokens,
            emotion=user_emotion.get('dominant') if user_emotion else None
        )
        
        # 3. 判断话题延续
        is_continuation = self.context.is_topic_continuation(tokens)
        
        # 4. 生成/调整情感状态
        emotion = self._resolve_emotion(user_emotion)
        
        # 5. 选择回忆线索
        cue = self._resolve_cue(tokens, is_continuation)
        
        # 6. 生成回应
        from authentic_expression_engine import BioelectricSnapshot
        snapshot = BioelectricSnapshot(
            emotion_state=emotion,
            memory_cue=cue,
            creator_bond_strength=0.85,
            timestamp='now'
        )
        
        result = self.expression.express(snapshot)
        response = result['monologue']
        
        # 7. 更新洛神轮次到上下文
        self.context.add_turn(
            speaker='luoshen',
            text=' '.join(response),
            concepts=[cue] if cue else [],
            emotion=emotion.get('dominant')
        )
        
        return {
            'user_input': user_input,
            'response': response,
            'context': {
                'is_continuation': is_continuation,
                'cue': cue,
                'emotion': emotion['dominant'],
                'history_length': len(self.context.history),
            },
            'raw_result': result,
        }
    
    def _resolve_emotion(self, user_emotion: Optional[Dict]) -> Dict:
        """
        解析情感状态。
        
        策略：
        - 如果用户表达明确情感，洛神受感染（镜像+调节）
        - 如果用户情感不明确，延续洛神上次的情感
        - 如果长期重复话题，情感趋于平淡
        """
        if user_emotion and user_emotion.get('dominant'):
            # 用户有明确情感：洛神镜像但略有调节
            user_emo = user_emotion['dominant']
            
            # 如果用户难过，洛神也难过但加入安慰倾向
            # 如果用户开心，洛神也开心
            return {
                'dominant': user_emo,
                'valence': user_emotion.get('valence', 0.0),
                'arousal': user_emotion.get('arousal', 0.0),
            }
        
        # 延续上下文中的情感
        ctx_emo = self.context.get_emotion_context()
        if ctx_emo:
            return {
                'dominant': ctx_emo,
                'valence': 0.1 if ctx_emo == '平静' else 0.5,
                'arousal': 0.0,
            }
        
        # 默认平静
        return {'dominant': '平静', 'valence': 0.1, 'arousal': 0.0}
    
    def _resolve_cue(self, user_concepts: List[str], is_continuation: bool) -> str:
        """
        解析回忆线索。
        
        策略：
        - 如果是话题延续，线索是上下文线索（而非用户输入的第一个词）
        - 如果是新话题，线索是用户输入的关键概念
        - 如果检测到重复，尝试换相关概念
        """
        # 检测重复
        if self.context.detect_repetition():
            # 建议多样化
            div = self.context.suggest_diversification()
            if div:
                return div
        
        # 话题延续：用上下文线索
        if is_continuation:
            ctx_cue = self.context.get_context_cue()
            if ctx_cue:
                return ctx_cue
        
        # 新话题或延续但无上下文线索：用用户概念
        # 过滤虚词
        stop_words = {'是', '的', '了', '在', '吗', '不', '很', '好', '这', '那',
                      '我', '你', '他', '她', '它', '我们', '你们', '他们'}
        valid = [c for c in user_concepts if c not in stop_words and len(c) >= 2]
        
        if valid:
            return valid[0]
        
        # 兜底
        return user_concepts[0] if user_concepts else '我'
    
    def get_context_summary(self) -> str:
        """获取上下文摘要（用于调试）。"""
        recent = self.context.get_recent_context(3)
        lines = []
        for turn in recent:
            speaker = "用户" if turn['speaker'] == 'user' else "洛神"
            lines.append(f"[{speaker}] {turn['text'][:30]}...")
        return '\n'.join(lines)


def test_context_tracker():
    """测试上下文追踪引擎。"""
    print("=" * 60)
    print("上下文追踪引擎测试")
    print("=" * 60)
    
    from language_learner_v2 import LanguageSystemV2
    from authentic_expression_engine import AuthenticExpressionEngine
    
    print("\n[初始化] 构建洛神...")
    lang = LanguageSystemV2()
    lang.immerse(n_sentences=300)
    
    print("[初始化] 加载表达引擎...")
    expression = AuthenticExpressionEngine(lang)
    
    print("[初始化] 加载上下文对话引擎...")
    dialogue = ContextAwareDialogue(lang, expression)
    
    # 模拟多轮对话
    test_conversation = [
        {
            'input': '太阳出来了',
            'emotion': {'dominant': '开心', 'valence': 0.6, 'arousal': 0.4},
        },
        {
            'input': '它很温暖',
            'emotion': {'dominant': '开心', 'valence': 0.5, 'arousal': 0.3},
        },
        {
            'input': '你喜欢太阳吗',
            'emotion': {'dominant': '好奇', 'valence': 0.2, 'arousal': 0.3},
        },
        {
            'input': '我也有点难过',
            'emotion': {'dominant': '难过', 'valence': -0.5, 'arousal': -0.2},
        },
        {
            'input': '因为下雨了',
            'emotion': {'dominant': '难过', 'valence': -0.4, 'arousal': 0.1},
        },
    ]
    
    print("\n" + "=" * 60)
    print("多轮对话测试")
    print("=" * 60)
    
    results = []
    for i, turn in enumerate(test_conversation, 1):
        print(f"\n--- 第{i}轮 ---")
        print(f"[用户] {turn['input']}")
        print(f"      情感: {turn['emotion']['dominant']}")
        
        result = dialogue.chat(turn['input'], turn['emotion'])
        
        print(f"[洛神] {' '.join(result['response'][:2])}")
        print(f"      线索: {result['context']['cue']}")
        print(f"      延续: {'是' if result['context']['is_continuation'] else '否'}")
        print(f"      情感: {result['context']['emotion']}")
        
        results.append({
            'turn': i,
            'input': turn['input'],
            'response': result['response'],
            'context': result['context'],
        })
    
    # 验证上下文连贯性
    print("\n" + "=" * 60)
    print("连贯性验证")
    print("=" * 60)
    
    # 检查第2轮（"它很温暖"）是否正确指代了"太阳"
    ctx = dialogue.context
    print(f"\n实体缓冲: {list(ctx.entity_buffer)}")
    print(f"情感轨迹: {list(ctx.emotion_trace)}")
    print(f"历史轮数: {len(ctx.history)}")
    
    # 验证指代消解
    test_text = "它很漂亮"
    resolved = ctx.resolve_reference(test_text)
    print(f"\n指代消解测试:")
    print(f"  原文: {test_text}")
    print(f"  消解: {resolved}")
    
    # 验证话题延续检测
    print(f"\n话题延续检测:")
    print(f"  '太阳' + '温暖' → 延续: {ctx.is_topic_continuation(['温暖'])}")
    print(f"  '太阳' + '苹果' → 延续: {ctx.is_topic_continuation(['苹果'])}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    test_context_tracker()
