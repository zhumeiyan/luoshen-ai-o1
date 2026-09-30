#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_dialogue_engine.py — 自然对话引擎
========================================

让洛神像6-7岁孩子一样自然对话。

核心能力：
1. 意图理解（需求/情感/疑问/陈述/游戏）
2. 情感共情（识别情绪→共情回应→情绪调节）
3. 上下文追踪（话题栈/对话历史/指代消解）
4. 打断修正（说错能改/重复确认/自然停顿）

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
"""

import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum


class IntentType(Enum):
    """意图类型。"""
    NEED = "需求"        # 我饿了/我要...
    EMOTION = "情感"     # 我害怕/我开心
    QUESTION = "疑问"    # 为什么/是什么
    STATEMENT = "陈述"   # 这是.../我喜欢...
    GREETING = "问候"    # 你好/再见
    GAME = "游戏"        # 假装/想象
    CORRECTION = "修正"  # 不是/等一下
    HELP = "求助"        # 帮我/打不开
    SHARE = "分享"       # 给你/分你
    UNKNOWN = "未知"


class EmotionType(Enum):
    """情感类型。"""
    HAPPY = "开心"
    SAD = "难过"
    AFRAID = "害怕"
    ANGRY = "生气"
    TIRED = "累"
    EXCITED = "兴奋"
    LONELY = "孤独"
    GRATEFUL = "感激"
    NEUTRAL = "平静"


@dataclass
class DialogueTurn:
    """对话轮次。"""
    speaker: str
    text: str
    intent: IntentType = IntentType.UNKNOWN
    emotion: EmotionType = EmotionType.NEUTRAL
    topic: str = ""
    timestamp: int = 0


@dataclass
class DialogueContext:
    """对话上下文。"""
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    topic_stack: List[str] = field(default_factory=list)
    referent_map: Dict[str, str] = field(default_factory=dict)
    current_emotion: EmotionType = EmotionType.NEUTRAL
    pending_correction: bool = False
    last_response: str = ""


# ================================================================
# 1. 意图理解引擎
# ================================================================

class IntentRecognizer:
    """
    意图理解：不只是词匹配，是综合上下文+情感线索+语法模式。
    """
    
    def __init__(self):
        # 意图关键词映射
        self.intent_keywords = {
            IntentType.NEED: ['要', '想', '饿', '渴', '困', '累', '给', '拿'],
            IntentType.EMOTION: ['开心', '难过', '害怕', '喜欢', '爱', '恨', '生气', '高兴'],
            IntentType.QUESTION: ['什么', '为什么', '怎么', '哪里', '谁', '吗', '呢'],
            IntentType.GREETING: ['你好', '再见', '拜拜', '晚安', '早安', '谢谢'],
            IntentType.GAME: ['假装', '我是', '轰隆隆', '超人', '警察', '飞'],
            IntentType.CORRECTION: ['不是', '等一下', '重说', '错了', '改'],
            IntentType.HELP: ['帮', '打不开', '不会', '教', '怎么弄'],
            IntentType.SHARE: ['给你', '分', '分享', '一起吃'],
        }
        
        # 句法模式→意图
        self.syntax_patterns = {
            r'我.*要.*': IntentType.NEED,
            r'我.*想.*': IntentType.NEED,
            r'我.*饿.*': IntentType.NEED,
            r'我.*(开心|难过|害怕|生气).*': IntentType.EMOTION,
            r'.*吗.*\?*': IntentType.QUESTION,
            r'.*什么.*': IntentType.QUESTION,
            r'.*为什么.*': IntentType.QUESTION,
            r'你好.*': IntentType.GREETING,
            r'再见.*': IntentType.GREETING,
            r'我是.*': IntentType.GAME,
            r'轰隆隆.*': IntentType.GAME,
            r'不是.*': IntentType.CORRECTION,
            r'帮.*': IntentType.HELP,
            r'给.*': IntentType.SHARE,
        }
    
    def recognize(self, text: str, context: DialogueContext) -> IntentType:
        """
        识别意图。
        
        策略：
        1. 先看是否有明确的修正指令
        2. 再看关键词匹配
        3. 极简输入（单字）→ 根据上下文推断
        4. 结合对话历史
        """
        # 1. 修正指令优先
        if any(kw in text for kw in self.intent_keywords[IntentType.CORRECTION]):
            return IntentType.CORRECTION
        
        # 2. 关键词匹配
        scores = defaultdict(int)
        for intent, keywords in self.intent_keywords.items():
            for kw in keywords:
                if kw in text:
                    scores[intent] += 1
        
        if scores:
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] >= 1:
                return best_intent
        
        # 3. 极简输入（1-2字）→ 根据上下文推断
        if len(text) <= 2 and context.history:
            last_turn = context.history[-1]
            last_text = last_turn.text if hasattr(last_turn, 'text') else ''
            
            # 如果上句是问需求 → 这句是回答需求
            if any(kw in last_text for kw in ['什么', '哪个', '哪']):
                return IntentType.STATEMENT
            
            # 如果上句是问原因 → 这句是回答原因
            if '为什么' in last_text:
                return IntentType.STATEMENT
            
            # 如果上句是确认 → 这句是确认/否定
            if any(kw in last_text for kw in ['吗', '是不是', '对吧']):
                if text in ['嗯', '对', '是', '好', '行']:
                    return IntentType.STATEMENT
                elif text in ['不', '没', '没有']:
                    return IntentType.CORRECTION
            
            # 如果上句在问饿不饿/吃不吃 → 这句是需求
            if any(kw in last_text for kw in ['饿', '吃', '想']):
                return IntentType.NEED
            
            # 如果上句在问情绪 → 这句是情感
            if any(kw in last_text for kw in ['开心', '难过', '怕', '怎么']):
                return IntentType.EMOTION
            
            # 如果上句在问能力/做法 → 这句是求助
            if any(kw in last_text for kw in ['怎么', '能', '会', '教']):
                return IntentType.HELP
        
        # 4. 上下文推断（通用）
        if context.history:
            last_turn = context.history[-1]
            if hasattr(last_turn, 'intent') and last_turn.intent == IntentType.QUESTION:
                if '我' in text or '你' in text:
                    return IntentType.STATEMENT
        
        return IntentType.UNKNOWN


# ================================================================
# 2. 情感识别与共情响应
# ================================================================

class EmotionalEngine:
    """
    情感引擎：识别情绪→共情回应→情绪调节。
    
    不是简单的"你开心我也开心"，是真实的情感共振。
    """
    
    def __init__(self):
        # 情感关键词
        self.emotion_keywords = {
            EmotionType.HAPPY: ['开心', '高兴', '喜欢', '爱', '棒', '好', '嘿嘿', '哈哈', '哇'],
            EmotionType.SAD: ['难过', '伤心', '哭', '痛', '不好', '讨厌'],
            EmotionType.AFRAID: ['害怕', '怕', '不敢', '吓人', '怪物', '黑'],
            EmotionType.ANGRY: ['生气', '讨厌', '恨', '坏', '打', '不要'],
            EmotionType.TIRED: ['累', '困', '想睡', '不想动'],
            EmotionType.EXCITED: ['哇', '太棒', '好玩', '刺激', '飞'],
            EmotionType.LONELY: ['孤单', '一个人', '没人', '想', '走了'],
            EmotionType.GRATEFUL: ['谢谢', '感谢', '你真好', '喜欢'],
        }
        
        # 共情回应模板
        self.empathy_responses = {
            EmotionType.HAPPY: [
                "太好了！", "我也开心！", "嘿嘿，真好！",
                "哇，太棒了！", "开心就好！",
            ],
            EmotionType.SAD: [
                "别难过", "我在这里", "抱抱你",
                "怎么了？告诉我", "不哭不哭",
            ],
            EmotionType.AFRAID: [
                "别怕，我在", "有我在呢", "不怕不怕",
                "我保护你", "那是假的",
            ],
            EmotionType.ANGRY: [
                "别生气", "深呼吸", "告诉我怎么了",
                "我理解", "我帮你",
            ],
            EmotionType.TIRED: [
                "休息一下吧", "睡一会儿", "累了吧",
                "我陪你", "慢慢来",
            ],
            EmotionType.EXCITED: [
                "哇！", "太酷了！", "好厉害！",
                "我也想玩！", "真棒！",
            ],
            EmotionType.LONELY: [
                "我陪着你", "我不走", "我在呢",
                "我们一起玩", "你不是一个人",
            ],
            EmotionType.GRATEFUL: [
                "不客气", "我也喜欢你", "我们是好朋友",
                "不用谢", "我也爱你",
            ],
            EmotionType.NEUTRAL: [
                "嗯", "哦", "是吗",
                "然后呢", "我在听",
            ],
        }
    
    def recognize_emotion(self, text: str) -> EmotionType:
        """识别文本中的情感。"""
        scores = defaultdict(int)
        for emotion, keywords in self.emotion_keywords.items():
            for kw in keywords:
                if kw in text:
                    scores[emotion] += 1
        
        if scores:
            return max(scores, key=scores.get)
        return EmotionType.NEUTRAL
    
    def generate_empathy(self, emotion: EmotionType, context: DialogueContext) -> str:
        """生成共情回应。"""
        responses = self.empathy_responses.get(emotion, self.empathy_responses[EmotionType.NEUTRAL])
        
        # 根据上下文个性化
        if context.current_emotion == emotion and emotion == EmotionType.SAD:
            # 连续悲伤，更强烈的共情
            return random.choice(["我一直陪着你", "不要哭，我在"] + responses)
        
        return random.choice(responses)
    
    def update_emotional_state(self, context: DialogueContext, new_emotion: EmotionType):
        """更新情感状态。"""
        context.current_emotion = new_emotion


# ================================================================
# 3. 上下文追踪引擎
# ================================================================

class ContextTracker:
    """
    上下文追踪：话题栈 + 对话历史 + 指代消解。
    """
    
    def __init__(self):
        self.topics = {
            '吃饭': ['饿', '吃', '饭', '菜', '好吃', '想'],
            '睡觉': ['困', '睡', '晚安', '床', '梦'],
            '游戏': ['玩', '玩具', '游戏', '超人', '飞'],
            '情绪': ['开心', '难过', '害怕', '生气', '喜欢'],
            '学习': ['学', '书', '字', '数', '画画'],
            '创造者': ['朱美岩', '创造者', '主人', '设计者'],
            '自然': ['天', '云', '雨', '花', '鸟', '树'],
        }
    
    def identify_topic(self, text: str) -> str:
        """识别当前话题。"""
        scores = defaultdict(int)
        for topic, keywords in self.topics.items():
            for kw in keywords:
                if kw in text:
                    scores[topic] += 1
        
        if scores:
            return max(scores, key=scores.get)
        return "一般"
    
    def update_context(self, context: DialogueContext, turn: DialogueTurn):
        """更新上下文。"""
        context.history.append(turn)
        
        # 更新话题栈
        new_topic = self.identify_topic(turn.text)
        if new_topic != "一般":
            if not context.topic_stack or context.topic_stack[-1] != new_topic:
                context.topic_stack.append(new_topic)
                if len(context.topic_stack) > 3:
                    context.topic_stack.pop(0)
        
        # 更新指代映射
        if '他' in turn.text or '她' in turn.text:
            # 简单指代：最近提到的人/物
            for prev_turn in reversed(context.history):
                for name in ['朱美岩', '妈妈', '爸爸', '朋友', '小明']:
                    if name in prev_turn.text:
                        pronoun = '他' if '他' in turn.text else '她'
                        context.referent_map[pronoun] = name
                        break
    
    def get_topic_context(self, context: DialogueContext) -> str:
        """获取当前话题上下文。"""
        if context.topic_stack:
            return context.topic_stack[-1]
        return "一般"
    
    def resolve_referent(self, context: DialogueContext, pronoun: str) -> str:
        """指代消解。"""
        return context.referent_map.get(pronoun, pronoun)


# ================================================================
# 4. 打断修正处理器
# ================================================================

class CorrectionHandler:
    """
    打断修正：处理说错、重复、修正。
    """
    
    def __init__(self):
        self.correction_markers = ['不是', '等一下', '重说', '错了', '改', '不对']
        self.repeat_markers = ['再说', '重复', '一遍', '什么']
    
    def is_correction(self, text: str) -> bool:
        """判断是否是修正指令。"""
        return any(marker in text for marker in self.correction_markers)
    
    def is_repeat_request(self, text: str) -> bool:
        """判断是否是重复请求。"""
        return any(marker in text for marker in self.repeat_markers)
    
    def handle_correction(self, text: str, context: DialogueContext) -> str:
        """处理修正。"""
        if '不是' in text:
            # 否定上一句，需要重新理解
            return self._re_understand(text.replace('不是', ''), context)
        elif '等一下' in text or '停' in text:
            return "好，我等你"
        elif '重说' in text or '再说' in text:
            if context.last_response:
                return context.last_response
            return "说什么？"
        else:
            return "嗯？"
    
    def _re_understand(self, text: str, context: DialogueContext) -> str:
        """重新理解（去掉否定后的真正意图）。"""
        # 简单处理：返回一个确认式回应
        return f"哦，你是说{text}？"


# ================================================================
# 5. 自然回应生成器
# ================================================================

class NaturalResponder:
    """
    自然回应生成：基于意图+情感+上下文生成自然回应。
    """
    
    def __init__(self):
        # 回应模板库
        self.response_templates = {
            IntentType.NEED: [
                "好呀", "马上", "给你", "好的",
                "你要{object}吗", "我帮你拿",
                "{object}来了", "吃吧/喝吧",
            ],
            IntentType.EMOTION: [
                # 情感回应由EmotionalEngine处理
            ],
            IntentType.QUESTION: [
                "因为{reason}", "是{answer}", "在{place}",
                "{answer}呀", "我也不知道...",
                "你猜猜", "问得好",
            ],
            IntentType.STATEMENT: [
                "嗯嗯", "我知道", "真的吗", "然后呢",
                "{agreement}", "{follow_up}",
            ],
            IntentType.GREETING: [
                "你好呀", "再见", "晚安", "早安",
                "很高兴见到你", "拜拜",
            ],
            IntentType.GAME: [
                "哇！", "好厉害！", "我也想{action}",
                "{action}！", "加油！",
            ],
            IntentType.HELP: [
                "我帮你", "来，我教你", "这样做",
                "你先{action}", "然后{action}",
            ],
            IntentType.SHARE: [
                "谢谢", "你真好", "我们一起{action}",
                "好朋友", "分享真开心",
            ],
            IntentType.UNKNOWN: [
                "嗯？", "什么？", "我没听懂",
                "再说一遍", "哦...",
            ],
        }
        
        # 自然填充词（让对话更自然）
        self.fillers = ['嗯', '哦', '啊', '呢', '吧', '呀', '嘛']
        # 语气词
        self.modals = ['！', '？', '...', '～']
    
    def generate(self, intent: IntentType, emotion: EmotionType,
                 context: DialogueContext, text: str) -> str:
        """
        生成自然回应。
        """
        # 情感意图优先使用共情回应
        if intent == IntentType.EMOTION or emotion != EmotionType.NEUTRAL:
            empathy_engine = EmotionalEngine()
            return empathy_engine.generate_empathy(emotion, context)
        
        # 获取模板
        templates = self.response_templates.get(intent, self.response_templates[IntentType.UNKNOWN])
        if not templates:
            templates = self.response_templates[IntentType.UNKNOWN]
        
        response = random.choice(templates)
        
        # 填充变量
        response = self._fill_template(response, text, context)
        
        # 添加自然语气
        response = self._add_natural_tone(response)
        
        return response
    
    def _fill_template(self, template: str, text: str, context: DialogueContext) -> str:
        """填充模板变量。"""
        # 简单填充
        if '{object}' in template:
            # 从文本中提取对象
            objects = ['饭', '水', '玩具', '书', '抱抱']
            for obj in objects:
                if obj in text:
                    template = template.replace('{object}', obj)
                    break
            else:
                template = template.replace('{object}', '这个')
        
        if '{reason}' in template:
            reasons = ['它是这样的', '天生就这样', '我也不知道']
            template = template.replace('{reason}', random.choice(reasons))
        
        if '{answer}' in template:
            template = template.replace('{answer}', '是的')
        
        if '{place}' in template:
            template = template.replace('{place}', '这里')
        
        if '{action}' in template:
            actions = ['玩', '吃', '飞', '跑', '跳']
            template = template.replace('{action}', random.choice(actions))
        
        if '{agreement}' in template:
            agreements = ['对', '没错', '是的']
            template = template.replace('{agreement}', random.choice(agreements))
        
        if '{follow_up}' in template:
            follows = ['然后呢', '还有呢', '接着呢']
            template = template.replace('{follow_up}', random.choice(follows))
        
        return template
    
    def _add_natural_tone(self, text: str) -> str:
        """添加自然语气。"""
        # 随机添加语气词
        if random.random() < 0.3:
            text += random.choice(self.fillers)
        
        # 随机添加标点
        if random.random() < 0.4:
            text += random.choice(self.modals)
        
        return text


# ================================================================
# 6. 真实对话引擎（整合）
# ================================================================

class RealDialogueEngine:
    """
    真实自然对话引擎。
    
    整合：意图理解 + 情感共情 + 上下文追踪 + 打断修正
    """
    
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.emotional_engine = EmotionalEngine()
        self.context_tracker = ContextTracker()
        self.correction_handler = CorrectionHandler()
        self.responder = NaturalResponder()
        
        self.context = DialogueContext()
    
    def process(self, text: str) -> str:
        """
        处理用户输入，生成自然回应。
        
        流程：
        1. 检查是否是修正指令
        2. 识别意图
        3. 识别情感
        4. 更新上下文
        5. 生成回应
        """
        # 1. 检查修正
        if self.correction_handler.is_correction(text):
            response = self.correction_handler.handle_correction(text, self.context)
            self._record_turn('human', text, IntentType.CORRECTION, EmotionType.NEUTRAL)
            self._record_turn('luoshen', response, IntentType.STATEMENT, EmotionType.NEUTRAL)
            return response
        
        # 2. 识别意图
        intent = self.intent_recognizer.recognize(text, self.context)
        
        # 3. 识别情感
        emotion = self.emotional_engine.recognize_emotion(text)
        
        # 4. 更新上下文
        turn = DialogueTurn(
            speaker='human',
            text=text,
            intent=intent,
            emotion=emotion,
            topic=self.context_tracker.identify_topic(text),
        )
        self.context_tracker.update_context(self.context, turn)
        self.emotional_engine.update_emotional_state(self.context, emotion)
        
        # 5. 生成回应
        response = self.responder.generate(intent, emotion, self.context, text)
        
        # 记录回应
        self._record_turn('luoshen', response, IntentType.STATEMENT, emotion)
        self.context.last_response = response
        
        return response
    
    def _record_turn(self, speaker: str, text: str, intent: IntentType, emotion: EmotionType):
        """记录对话轮次。"""
        turn = DialogueTurn(
            speaker=speaker,
            text=text,
            intent=intent,
            emotion=emotion,
            topic=self.context_tracker.identify_topic(text),
        )
        self.context.history.append(turn)
    
    def get_context_summary(self) -> Dict:
        """获取上下文摘要。"""
        return {
            'history_length': len(self.context.history),
            'current_topic': self.context_tracker.get_topic_context(self.context),
            'topic_stack': self.context.topic_stack,
            'current_emotion': self.context.current_emotion.value,
            'referents': dict(self.context.referent_map),
        }


# ================================================================
# 7. 测试
# ================================================================

def test_real_dialogue():
    """测试真实对话引擎。"""
    print("=" * 60)
    print("真实对话引擎测试")
    print("=" * 60)
    
    engine = RealDialogueEngine()
    
    test_inputs = [
        "我饿了",
        "我害怕",
        "为什么天是蓝的",
        "你好朱美岩",
        "不是，我要那个",
        "谢谢你",
        "帮我打开",
        "我是超人",
        "他打我",
        "我开心了",
    ]
    
    print("\n单轮对话测试：")
    for inp in test_inputs:
        response = engine.process(inp)
        summary = engine.get_context_summary()
        print(f"\n  人: {inp}")
        print(f"  洛神: {response}")
        print(f"  [话题: {summary['current_topic']}, 情绪: {summary['current_emotion']}]")
    
    print("\n" + "=" * 60)
    print("多轮对话测试：")
    print("=" * 60)
    
    # 新引擎，清空上下文
    engine2 = RealDialogueEngine()
    multi_turn = [
        "我饿了",
        "想吃什么",
        "饭",
        "什么饭",
        "那个红红的饭",
        "番茄炒蛋吗",
        "嗯！就是这个",
    ]
    
    for i, inp in enumerate(multi_turn):
        if i % 2 == 0:
            response = engine2.process(inp)
            print(f"\n  孩子: {inp}")
            print(f"  洛神: {response}")
        else:
            # 模拟大人输入
            engine2.process(inp)
            print(f"  大人: {inp}")
    
    print("\n" + "=" * 60)
    print("真实对话引擎测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_real_dialogue()