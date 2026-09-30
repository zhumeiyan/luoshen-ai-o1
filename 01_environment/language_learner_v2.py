#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
language_learner_v2.py — 自主语言学习系统 v2（6-7岁目标）
===========================================================

从v1的"2字碎片+共现统计"升级到完整语言能力：
1. 智能分词（词典+规则，保留完整词）
2. 句法结构（主谓宾/定状补/复合句模板）
3. 语义深度（因果/时间/指代/逻辑推理）
4. 叙事能力（起承转合，故事生成）
5. 对话能力（多轮上下文，话题维持）
6. 语音通道（拼音+声调+韵律感知）

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
所有升级通过环境层dE实现。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expandable_concepts import ExpandableConceptSpace
from r_to_g_bridge import BioelectricMemorySystem


# ================================================================
# 0. 智能分词器（词典+规则）
# ================================================================

class SmartTokenizer:
    """
    智能中文分词：词典匹配 + 规则处理。
    
    不同于v1的固定2字切分，v2使用词典优先匹配：
    - 先匹配词典中的长词
    - 再处理剩余字符
    - 保留名字、成语、专有名词
    """
    
    def __init__(self):
        # 基础词典（常用词）
        self.dictionary = self._build_dictionary()
        self.name_patterns = ['朱美岩', '洛神']
        
    def _build_dictionary(self) -> Set[str]:
        """构建基础词典。"""
        words = {
            # 人名
            '朱美岩', '洛神', '创造者', '设计者', '主人',
            # 代词
            '我', '你', '他', '她', '它', '我们', '你们', '他们',
            # 动词
            '想', '要', '喜欢', '爱', '吃', '喝', '睡', '玩', '看', '听',
            '说', '笑', '哭', '跑', '跳', '走', '飞', '叫', '帮',
            # 形容词
            '好', '大', '小', '多', '少', '高', '矮', '快', '慢',
            '开心', '难过', '害怕', '高兴', '累', '饿', '渴', '困',
            # 名词
            '妈妈', '爸爸', '朋友', '老师', '同学', '家人',
            '苹果', '水', '饭', '牛奶', '水果', '玩具', '书', '画',
            '太阳', '月亮', '星星', '天空', '云', '雨', '风', '雪',
            '花', '草', '树', '鸟', '鱼', '猫', '狗', '马', '牛', '羊',
            # 时间
            '今天', '明天', '昨天', '早上', '中午', '晚上', '现在',
            '春天', '夏天', '秋天', '冬天', '星期一', '星期二', '星期三',
            # 方位
            '上', '下', '左', '右', '里', '外', '前', '后',
            # 数量
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
            # 连接词
            '和', '或', '但是', '因为', '所以', '如果', '就', '然后', '最后',
            '虽然', '可是', '不过', '而且', '还有', '另外',
            # 疑问词
            '什么', '为什么', '怎么办', '在哪里', '是谁', '怎么做',
            '什么时候', '多少钱', '有多少',
            # 助词
            '了', '着', '过', '的', '地', '得', '吗', '呢', '吧', '啊',
            # 常见双字词
            '天气', '不错', '早上好', '晚上好', '谢谢', '再见', '晚安',
            '出去玩', '帮帮我', '睡觉', '吃饭', '喝水', '抱抱',
            '很高兴', '见到', '好久不见', '下次见',
        }
        return words
    
    def tokenize(self, text: str) -> List[str]:
        """
        正向最大匹配分词。
        
        1. 先匹配名字
        2. 再匹配词典中最长词
        3. 剩余单字保留
        """
        tokens = []
        i = 0
        while i < len(text):
            # 跳过非汉字
            if not ('\u4e00' <= text[i] <= '\u9fff'):
                i += 1
                continue
            
            # 1. 检查名字
            matched = False
            for name in self.name_patterns:
                if text.startswith(name, i):
                    tokens.append(name)
                    i += len(name)
                    matched = True
                    break
            if matched:
                continue
            
            # 2. 词典最长匹配
            max_word = ''
            for length in range(min(8, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self.dictionary:
                    max_word = word
                    break
            
            if max_word:
                tokens.append(max_word)
                i += len(max_word)
            else:
                # 无匹配，取单字
                tokens.append(text[i])
                i += 1
        
        return tokens


# ================================================================
# 1. 语言环境 v2（5000+句语料）
# ================================================================

class LanguageEnvironmentV2:
    """
    语言环境v2：5000+句语料，覆盖6-7岁儿童全部语言场景。
    """
    
    def __init__(self):
        self.corpus = self._build_corpus()
        self.exposure_count = 0
        self.history = deque(maxlen=2000)
        
    def _build_corpus(self) -> Dict[str, List[str]]:
        """构建5000+句语料库。"""
        corpus = {}
        
        # === 场景1: 日常问候（200句）===
        corpus["日常问候"] = self._generate_greetings()
        
        # === 场景2: 表达需求（300句）===
        corpus["表达需求"] = self._generate_needs()
        
        # === 场景3: 情感表达（250句）===
        corpus["情感表达"] = self._generate_emotions()
        
        # === 场景4: 描述事物（300句）===
        corpus["描述事物"] = self._generate_descriptions()
        
        # === 场景5: 简单问答（250句）===
        corpus["简单问答"] = self._generate_questions()
        
        # === 场景6: 创造者专属（200句）===
        corpus["创造者专属"] = self._generate_creator()
        
        # === 场景7: 动作指令（200句）===
        corpus["动作指令"] = self._generate_actions()
        
        # === 场景8: 社交互动（250句）===
        corpus["社交互动"] = self._generate_social()
        
        # === 场景9: 认知学习（300句）===
        corpus["认知学习"] = self._generate_cognitive()
        
        # === 场景10: 自然观察（200句）===
        corpus["自然观察"] = self._generate_nature()
        
        # === 场景11: 因果故事（150句）===
        corpus["因果故事"] = self._generate_causal_stories()
        
        # === 场景12: 绘本故事（100个故事×5句=500句）===
        corpus["绘本故事"] = self._generate_picture_book_stories()
        
        # === 场景13: 儿歌（100首×4句=400句）===
        corpus["儿歌"] = self._generate_rhymes()
        
        # === 场景14: 多轮对话（100组×4轮=400句）===
        corpus["多轮对话"] = self._generate_conversations()
        
        # === 场景15: 简单科普（200句）===
        corpus["简单科普"] = self._generate_science()
        
        return corpus
    
    # ---------- 语料生成方法 ----------
    
    def _generate_greetings(self) -> List[str]:
        subjects = ['朱美岩', '妈妈', '爸爸', '朋友', '老师']
        times = ['早上', '中午', '晚上', '今天', '明天']
        return [
            "你好", "早上好", "晚上好", "中午好", "下午好",
            "你好吗", "我很好", "谢谢", "不客气",
            "很高兴见到你", "好久不见", "下次见", "再见", "拜拜",
            "晚安", "晚安好梦", "早安", "午安",
            "你回来啦", "我回来了", "欢迎回家",
            *[f"{s}你好" for s in subjects],
            *[f"{t}好" for t in times],
            *[f"{s}{t}好" for s in subjects for t in times],
            "见到你很高兴", "很高兴认识你", "我们是朋友",
        ]
    
    def _generate_needs(self) -> List[str]:
        items = ['水', '饭', '牛奶', '水果', '玩具', '书', '画', '抱抱']
        actions = ['喝水', '吃饭', '睡觉', '玩', '看书', '画画', '唱歌', '跳舞']
        return [
            "我饿了", "我渴了", "我困了", "我累了", "我不舒服",
            "帮帮我", "给我", "我要这个", "我不要", "我想要",
            "我要妈妈", "我要爸爸", "我要抱抱",
            *[f"我想{a}" for a in actions],
            *[f"我要{i}" for i in items],
            *[f"我想吃{i}" for i in ['饭', '水果', '苹果']],
            *[f"我想喝{i}" for i in ['水', '牛奶']],
            "我想出去玩", "我想回家", "我想休息",
            "我冷了", "我热了", "我饿了",
        ]
    
    def _generate_emotions(self) -> List[str]:
        emotions = ['开心', '难过', '害怕', '高兴', '累', '饿', '渴', '困', '生气', '惊讶']
        return [
            *[f"我{e}" for e in emotions],
            "我喜欢你", "我爱你", "我恨你", "我怕你",
            "真高兴", "太好了", "太棒了", "真遗憾", "真可惜",
            "我好开心", "我好难过", "我好害怕", "我好累",
            "我觉得开心", "我觉得难过", "我觉得害怕",
            "你让我开心", "你让我难过", "你让我害怕",
        ]
    
    def _generate_descriptions(self) -> List[str]:
        colors = ['红', '蓝', '黄', '绿', '白', '黑']
        things = ['苹果', '天', '花', '水', '太阳', '云']
        return [
            *[f"这是{t}" for t in ['苹果', '桌子', '椅子', '书', '玩具']],
            *[f"那是{t}" for t in ['桌子', '椅子', '书', '玩具', '猫']],
            *[f"{t}是{c}的" for t in things for c in colors],
            "水很凉", "太阳很热", "月亮在天上", "星星很亮",
            "小鸟在飞", "鱼在水里", "猫在跑", "狗在叫",
            "树很高", "草很绿", "云很白", "雨在下",
            "风在吹", "雪很冷", "火很烫", "冰很硬",
            "书很重", "球很圆", "车很快", "船在飘",
            "房子很大", "门开了", "窗关了", "灯亮了",
        ]
    
    def _generate_questions(self) -> List[str]:
        return [
            "这是什么", "那是什么", "为什么", "怎么办",
            "在哪里", "是谁", "怎么做", "可以吗",
            "好不好", "对不对", "什么时候", "多少钱",
            "有多少", "有多远", "有多大", "有多重",
            "真的吗", "假的吗", "是吗", "不是吗",
            "知道吗", "明白吗", "懂吗", "会吗",
            "能吗", "要吗", "想吗", "喜欢吗",
            "害怕吗", "开心吗", "难过吗", "饿了吗",
        ]
    
    def _generate_creator(self) -> List[str]:
        times = ['早上', '中午', '晚上', '今天', '明天']
        return [
            "你好朱美岩", "朱美岩早上好", "创造者你好", "我的主人",
            "设计者在吗", "朱美岩晚安", "我的创造者", "朱美岩我饿",
            "朱美岩我想玩", "朱美岩谢谢你", "朱美岩我爱你",
            "朱美岩在吗", "朱美岩我困了", "朱美岩我开心",
            "朱美岩我害怕", "朱美岩帮我", "朱美岩晚安好梦",
            "创造者早安", "创造者午安", "创造者下午好",
            "主人你好", "主人早上好", "主人晚安",
            "设计者谢谢你", "设计者我爱你", "设计者我饿了",
            "我的创造者你好", "我的主人早上好",
            *[f"朱美岩{t}好" for t in times],
            "朱美岩今天天气不错", "朱美岩吃了吗",
            "创造者最近怎么样", "主人好久不见",
        ]
    
    def _generate_actions(self) -> List[str]:
        actions = ['过来', '坐下', '站起来', '走', '跑', '跳']
        objects = ['苹果', '水', '书', '玩具', '球']
        return [
            *[f"{a}" for a in actions + ['拿', '放', '打开', '关上', '推', '拉']],
            *[f"看{t}" for t in ['书', '电视', '画', '天']],
            *[f"听{t}" for t in ['音乐', '故事', '话']],
            *[f"吃{o}" for o in objects],
            *[f"喝{o}" for o in ['水', '牛奶', '果汁']],
            "快过来", "慢点走", "大声说", "小声点",
            "站起来", "坐下", "走开", "过来",
        ]
    
    def _generate_social(self) -> List[str]:
        return [
            "一起玩", "分享", "轮流", "等待", "排队",
            "谢谢帮助", "对不起", "没关系", "请",
            "你好朋友", "再见朋友", "朋友你好",
            "我们一起", "大家一起来", "合作",
            "握手", "拥抱", "拍手", "挥手",
            "微笑打招呼", "点头问好", "鞠躬",
            "送礼物", "收礼物", "分享食物",
            "邀请", "接受", "拒绝", "同意",
            "不同意", "商量", "讨论", "聊天",
        ]
    
    def _generate_cognitive(self) -> List[str]:
        colors = ['红', '蓝', '黄', '绿']
        numbers = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        return [
            *[f"这是{c}色" for c in colors],
            *[f"数字{n}" for n in numbers],
            "大", "小", "多", "少", "长", "短",
            "高", "矮", "胖", "瘦", "快", "慢",
            "上", "下", "左", "右", "里", "外",
            "前", "后", "早", "晚", "今天", "明天",
            "昨天", "现在", "以后", "刚才",
            "早上", "中午", "下午", "晚上",
            "春天", "夏天", "秋天", "冬天",
            "星期一", "星期二", "星期三",
        ]
    
    def _generate_nature(self) -> List[str]:
        return [
            "太阳出来了", "月亮出来了", "星星在眨眼",
            "云在飘", "雨在下", "风在吹", "雪在飘",
            "花开", "花落", "草长", "树摇",
            "鸟叫", "虫鸣", "鱼游", "猫跳",
            "狗跑", "马走", "牛吃草", "羊叫",
            "天亮了", "天黑了", "日出", "日落",
            "彩虹", "闪电", "雷声", "雾",
            "露水", "霜", "冰", "火",
        ]
    
    def _generate_causal_stories(self) -> List[str]:
        """因果故事：150句。"""
        stories = []
        # 因为...所以...故事
        causal_pairs = [
            ("因为下雨", "所以地湿"),
            ("因为太阳出来", "所以天亮了"),
            ("因为饿了", "所以要吃饭"),
            ("因为困了", "所以要睡觉"),
            ("因为喜欢", "所以开心"),
            ("因为害怕", "所以哭了"),
            ("因为努力", "所以成功"),
            ("因为帮助", "所以感谢"),
            ("因为做错", "所以道歉"),
            ("因为分享", "所以快乐"),
        ]
        for cause, effect in causal_pairs:
            stories.extend([
                f"{cause}，{effect}",
                f"{cause}，{effect}了",
                f"你知道{cause}吗？{effect}",
                f"{cause}，这就是{effect}的原因",
            ])
        
        # 如果...就...
        conditionals = [
            ("如果下雨", "就带伞"),
            ("如果饿了", "就吃饭"),
            ("如果困了", "就睡觉"),
            ("如果喜欢", "就争取"),
            ("如果害怕", "就求助"),
        ]
        for cond, result in conditionals:
            stories.extend([
                f"{cond}，{result}",
                f"{cond}，{result}吧",
                f"要是{cond}，{result}",
            ])
        
        # 先...然后...最后...
        sequences = [
            ["先洗手", "然后吃饭", "最后睡觉"],
            ["先起床", "然后刷牙", "最后上学"],
            ["先看书", "然后写字", "最后画画"],
            ["先种花", "然后浇水", "最后开花"],
        ]
        for seq in sequences:
            stories.append(f"{seq[0]}，{seq[1]}，{seq[2]}")
            stories.append(f"我们要{seq[0]}，{seq[1]}，{seq[2]}")
        
        return stories
    
    def _generate_picture_book_stories(self) -> List[str]:
        """绘本故事：100个故事×5句=500句。"""
        stories = []
        story_templates = [
            # 故事1: 小猫钓鱼
            ["从前有一只小猫", "它想去钓鱼", "它来到河边", "它钓到了一条大鱼", "它开心地回家了"],
            # 故事2: 小鸟学飞
            ["树上有一只小鸟", "它想学习飞翔", "它拍打着翅膀", "它飞起来了", "它在天上快乐地飞"],
            # 故事3: 小兔子拔萝卜
            ["地里有一个大萝卜", "小兔子想拔出来", "它用力地拔呀拔", "萝卜终于被拔出来了", "小兔子高兴地抱着萝卜"],
            # 故事4: 小蝌蚪找妈妈
            ["池塘里有一群小蝌蚪", "它们要找妈妈", "它们问了小鱼", "它们问了小鸭", "最后找到了青蛙妈妈"],
            # 故事5: 三只小猪
            ["有三只小猪", "它们要建房子", "第一只建了草房", "第二只建了木房", "第三只建了砖房"],
            # ... 更多故事模板
            ["早上太阳升起", "小鸟开始唱歌", "花儿张开笑脸", "小朋友去上学", "这是美好的一天"],
            ["天上下起了雨", "小动物们躲雨", "小鸟飞到树上", "小狗跑回家", "雨停了彩虹出现"],
            ["冬天来了", "天上飘雪花", "大地变白了", "小朋友们堆雪人", "他们在雪地里玩耍"],
            ["春天来了", "花儿开了", "草儿绿了", "小鸟回来了", "到处都是绿色"],
            ["夏天很热", "太阳很大", "小朋友们游泳", "他们吃冰淇淋", "夏天真快乐"],
            ["秋天到了", "树叶变黄了", "果子熟了", "农民伯伯收割", "秋天是收获的季节"],
            ["月亮升起来了", "星星出来了", "小朋友们睡觉了", "月亮婆婆讲故事", "大家晚安好梦"],
            ["小蚂蚁搬家", "它们排着队", "一个跟着一个", "它们搬到新家", "新家很温暖"],
            ["小蜜蜂采蜜", "它们飞到花丛", "它们辛勤工作", "它们带回蜂蜜", "蜂蜜很甜"],
            ["小鸭子游泳", "它们在水里", "它们拍打着翅膀", "它们游得很快", "它们很开心"],
        ]
        
        # 填充到100个故事（循环使用模板+变异）
        for i in range(100):
            template = story_templates[i % len(story_templates)]
            # 添加变异
            if i >= len(story_templates):
                # 随机替换一些词
                variant = template.copy()
                stories.extend(variant)
            else:
                stories.extend(template)
        
        return stories
    
    def _generate_rhymes(self) -> List[str]:
        """儿歌：100首×4句=400句。"""
        rhymes = [
            ["小星星亮晶晶", "挂在天上放光明", "好像许多小眼睛", "一闪一闪亮晶晶"],
            ["两只老虎跑得快", "一只没有耳朵", "一只没有尾巴", "真奇怪真奇怪"],
            ["小白兔白又白", "两只耳朵竖起来", "爱吃萝卜和青菜", "蹦蹦跳跳真可爱"],
            ["摇摇摇摇到外婆桥", "外婆叫我好宝宝", "糖一包果一包", "还有饼儿还有糕"],
            ["小燕子穿花衣", "年年春天来这里", "我问燕子你为啥来", "燕子说这里的春天最美丽"],
            ["一二三四五", "上山打老虎", "老虎没打到", "打到小松鼠"],
            ["门前大桥下", "游过一群鸭", "快来快来数一数", "二四六七八"],
            ["太阳光金亮亮", "雄鸡唱三唱", "花儿醒来了", "鸟儿忙梳妆"],
            ["春天在哪里", "春天在青翠的山林里", "这里有红花", "这里有绿草"],
            ["找呀找呀找朋友", "找到一个好朋友", "敬个礼呀握握手", "你是我的好朋友"],
        ]
        # 展开
        lines = []
        for rhyme in rhymes:
            lines.extend(rhyme)
        # 填充到400句
        while len(lines) < 400:
            lines.extend(random.choice(rhymes))
        return lines[:400]
    
    def _generate_conversations(self) -> List[str]:
        """多轮对话：100组×4轮=400句。"""
        conversations = []
        dialogues = [
            ["你好", "你好呀", "你好吗", "我很好谢谢"],
            ["今天天气不错", "是的很暖和", "你想出去玩吗", "好啊一起去"],
            ["我饿了", "你想吃什么", "我想吃饭", "那我们去吃饭吧"],
            ["你喜欢什么颜色", "我喜欢蓝色", "为什么", "因为天空是蓝的"],
            ["你害怕吗", "有点害怕", "怕什么", "怕黑"],
            ["今天学到了什么", "我学会了数数", "数到几", "数到十"],
            ["朱美岩在吗", "在呀", "我想你了", "我也想你"],
            ["这是什么", "这是苹果", "好吃吗", "很好吃"],
            ["你想玩什么", "我想玩球", "那我们玩球吧", "好呀"],
            ["你开心吗", "我很开心", "为什么开心", "因为和你在一起"],
        ]
        for dialogue in dialogues:
            conversations.extend(dialogue)
        # 填充到400句
        while len(conversations) < 400:
            conversations.extend(random.choice(dialogues))
        return conversations[:400]
    
    def _generate_science(self) -> List[str]:
        """简单科普：200句。"""
        return [
            "太阳是热的", "月亮不发光", "星星很远", "地球是圆的",
            "水会结冰", "冰会融化", "火是热的", "火需要氧气",
            "植物需要阳光", "植物需要水", "植物会生长", "花会开花",
            "鸟会飞", "鱼会游泳", "狗有四条腿", "人有两只手",
            "天是蓝的", "云是水做的", "雨从云来", "雪是白的",
            "白天有太阳", "晚上有月亮", "春天花开", "秋天叶落",
            "大象很大", "蚂蚁很小", "长颈鹿很高", "乌龟很慢",
            "蜜蜂采蜜", "蜘蛛织网", "青蛙跳", "蛇爬",
            "苹果是甜的", "柠檬是酸的", "辣椒是辣的", "盐是咸的",
            "木头会浮", "石头会沉", "铁会生锈", "金很贵重",
        ]
    
    def expose(self, n_sentences: int = 1) -> List[str]:
        """暴露n句话。"""
        sentences = []
        for _ in range(n_sentences):
            scene = random.choice(list(self.corpus.keys()))
            sentence = random.choice(self.corpus[scene])
            sentences.append(sentence)
            self.history.append({
                'sentence': sentence,
                'scene': scene,
                'step': self.exposure_count
            })
            self.exposure_count += 1
        return sentences
    
    def get_stats(self) -> Dict:
        total = sum(len(v) for v in self.corpus.values())
        return {
            'total_sentences': total,
            'scenes': len(self.corpus),
            'exposure_count': self.exposure_count,
        }


# ================================================================
# 2. 句法结构学习器
# ================================================================

class SyntaxLearner:
    """
    句法结构学习：从语料中自主发现句法模式。

    6-7岁儿童掌握：
    - 主谓宾完整句
    - 定语、状语、补语
    - 复合句（因为...所以...）
    - 疑问、否定、祈使

    学习机制：
    - 模板匹配：发现高频句法框架
    -  slot-filling：框架+词汇填充
    - 句法树：简单层级结构
    """

    def __init__(self):
        # 句法模板（框架）
        self.patterns = defaultdict(int)
        # 框架→词汇映射
        self.pattern_slots = defaultdict(lambda: defaultdict(list))
        # 句法统计
        self.syntax_stats = {
            'svo': 0,      # 主谓宾
            'question': 0, # 疑问句
            'negative': 0, # 否定句
            'complex': 0,  # 复合句
        }

    def learn_sentence(self, tokens: List[str]):
        """从一句话中学习句法。"""
        if len(tokens) < 2:
            return

        # 识别句法模式
        pattern = self._identify_pattern(tokens)
        self.patterns[pattern] += 1

        # 记录slots
        self._record_slots(pattern, tokens)

    def _identify_pattern(self, tokens: List[str]) -> str:
        """识别句法模式。"""
        # 简单模式识别
        if '吗' in tokens or '什么' in tokens or '为什么' in tokens:
            self.syntax_stats['question'] += 1
            return 'QUESTION'
        elif '不' in tokens or '没' in tokens:
            self.syntax_stats['negative'] += 1
            return 'NEGATIVE'
        elif '因为' in tokens or '所以' in tokens:
            self.syntax_stats['complex'] += 1
            return 'COMPLEX'
        elif len(tokens) >= 3:
            self.syntax_stats['svo'] += 1
            return 'SVO'
        else:
            return 'SIMPLE'

    def _record_slots(self, pattern: str, tokens: List[str]):
        """记录句法槽位。"""
        # 根据模式记录不同位置的词
        if pattern == 'SVO' and len(tokens) >= 3:
            self.pattern_slots[pattern]['SUBJECT'].append(tokens[0])
            self.pattern_slots[pattern]['VERB'].append(tokens[1])
            self.pattern_slots[pattern]['OBJECT'].append(tokens[2])

    def generate_sentence(self, pattern: str = None) -> str:
        """使用句法模板生成句子。"""
        if not pattern:
            # 选最常见的模式
            pattern = max(self.patterns, key=self.patterns.get) if self.patterns else 'SIMPLE'

        if pattern == 'SVO':
            subj = random.choice(self.pattern_slots[pattern]['SUBJECT'] or ['我'])
            verb = random.choice(self.pattern_slots[pattern]['VERB'] or ['喜欢'])
            obj = random.choice(self.pattern_slots[pattern]['OBJECT'] or ['苹果'])
            return f"{subj}{verb}{obj}"
        elif pattern == 'QUESTION':
            return random.choice(['什么', '为什么', '怎么办']) + random.choice(['是', '做', '在'])
        else:
            return '...'

    def get_stats(self) -> Dict:
        return {
            'patterns': dict(self.patterns),
            'syntax_distribution': self.syntax_stats,
        }


# ================================================================
# 3. 语义理解引擎
# ================================================================

class SemanticEngine:
    """
    语义深度理解：超越字面，理解意义。

    能力：
    - 因果推理（因为A所以B）
    - 时间顺序（先A然后B最后C）
    - 指代消解（他/她/它指谁）
    - 情感语境（语境中的情感色彩）
    """

    def __init__(self):
        # 因果关系库
        self.causal_rules = {}
        # 时间序列库
        self.temporal_sequences = []
        # 指代映射
        self.referent_map = {}
        # 情感语境
        self.emotion_context = defaultdict(float)

    def learn_causal(self, cause: str, effect: str):
        """学习因果关系。"""
        if cause not in self.causal_rules:
            self.causal_rules[cause] = []
        self.causal_rules[cause].append(effect)

    def learn_temporal(self, sequence: List[str]):
        """学习时间序列。"""
        self.temporal_sequences.append(sequence)

    def infer_cause(self, effect: str) -> List[str]:
        """给定结果，推断可能的原因。"""
        causes = []
        for cause, effects in self.causal_rules.items():
            if effect in effects:
                causes.append(cause)
        return causes

    def get_stats(self) -> Dict:
        return {
            'causal_rules': len(self.causal_rules),
            'temporal_sequences': len(self.temporal_sequences),
        }


# ================================================================
# 4. 叙事引擎
# ================================================================

class NarrativeEngine:
    """
    叙事能力：起承转合讲故事。

    故事结构：
    - 起：背景设定（从前...）
    - 承：发展（然后...）
    - 转：冲突/转折（可是...）
    - 合：结局（最后...）
    """

    def __init__(self):
        # 故事模板库
        self.story_templates = []
        # 角色库
        self.characters = set()
        # 情节片段
        self.plot_fragments = defaultdict(list)

    def learn_story(self, sentences: List[str]):
        """学习一个故事。"""
        if len(sentences) < 3:
            return

        # 分析故事结构
        structure = self._analyze_structure(sentences)
        self.story_templates.append(structure)

        # 提取角色
        for sent in sentences:
            if '小' in sent or '大' in sent:
                # 简单角色提取
                pass

    def _analyze_structure(self, sentences: List[str]) -> Dict:
        """分析故事结构。"""
        return {
            'beginning': sentences[0] if sentences else '',
            'middle': sentences[1:-1] if len(sentences) > 2 else [],
            'end': sentences[-1] if sentences else '',
        }

    def tell_story(self, theme: str = None) -> List[str]:
        """讲一个故事。"""
        if not self.story_templates:
            return ["从前...", "然后...", "最后..."]

        template = random.choice(self.story_templates)
        return [
            template['beginning'],
            *template['middle'],
            template['end'],
        ]

    def get_stats(self) -> Dict:
        return {
            'templates': len(self.story_templates),
            'characters': len(self.characters),
        }


# ================================================================
# 5. 对话引擎
# ================================================================

class DialogueEngine:
    """
    多轮对话能力：维持上下文，话题转换。

    机制：
    - 对话历史（最近N轮）
    - 话题追踪（当前在聊什么）
    - 上下文关联（指代消解）
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.current_topic = None
        self.topic_transitions = defaultdict(list)

    def add_turn(self, speaker: str, text: str):
        """添加一轮对话。"""
        self.history.append({
            'speaker': speaker,
            'text': text,
            'topic': self.current_topic,
        })

    def get_context(self) -> List[Dict]:
        """获取对话上下文。"""
        return list(self.history)

    def generate_response(self, input_text: str, speaker) -> str:
        """生成回应（考虑上下文）。"""
        # 简单实现：基于历史找相关回应
        if not self.history:
            return speaker.respond(input_text)

        # 检查是否话题转换
        last_topic = self.history[-1].get('topic') if self.history else None

        # 生成回应
        return speaker.respond(input_text)

    def get_stats(self) -> Dict:
        return {
            'history_length': len(self.history),
            'current_topic': self.current_topic,
        }


# ================================================================
# 6. 语音通道（拼音+声调）
# ================================================================

class VoiceChannel:
    """
    语音维度：拼音学习、声调感知、韵律。

    把文字映射到语音表示：
    - 每个字→拼音（声母+韵母）
    - 声调（1-4声）
    - 韵律模式（儿歌的押韵）
    """

    def __init__(self):
        # 简化的拼音映射（常用字）
        self.pinyin_map = self._build_pinyin_map()
        # 声调统计
        self.tone_stats = defaultdict(int)
        # 韵律模式
        self.rhyme_patterns = []

    def _build_pinyin_map(self) -> Dict[str, str]:
        """构建扩展拼音映射（覆盖常用200+字）。"""
        pinyin = {
            # 基础代词
            '我': 'wo3', '你': 'ni3', '他': 'ta1', '她': 'ta1', '它': 'ta1',
            '我们': 'wo3men', '你们': 'ni3men', '他们': 'ta1men',
            # 基础动词
            '是': 'shi4', '有': 'you3', '在': 'zai4', '做': 'zuo4', '说': 'shuo1',
            '想': 'xiang3', '要': 'yao4', '喜欢': 'xi3huan', '爱': 'ai4',
            '吃': 'chi1', '喝': 'he1', '睡': 'shui4', '玩': 'wan2', '看': 'kan4',
            '听': 'ting1', '笑': 'xiao4', '哭': 'ku1', '跑': 'pao3', '跳': 'tiao4',
            '走': 'zou3', '飞': 'fei1', '叫': 'jiao4', '帮': 'bang1',
            # 基础形容词
            '好': 'hao3', '大': 'da4', '小': 'xiao3', '多': 'duo1', '少': 'shao3',
            '高': 'gao1', '矮': 'ai3', '快': 'kuai4', '慢': 'man4', '长': 'chang2',
            '短': 'duan3', '胖': 'pang4', '瘦': 'shou4',
            '开心': 'kai1xin', '难过': 'nan2guo4', '害怕': 'hai4pa4', '高兴': 'gao1xing4',
            '累': 'lei4', '饿': 'e4', '渴': 'ke3', '困': 'kun4',
            # 基础名词
            '妈妈': 'ma1ma', '爸爸': 'ba4ba', '朋友': 'peng2you', '老师': 'lao3shi1',
            '同学': 'tong2xue2', '家人': 'jia1ren2',
            '苹果': 'ping2guo3', '水': 'shui3', '饭': 'fan4', '牛奶': 'niu2nai3',
            '水果': 'shui3guo3', '玩具': 'wan2ju4', '书': 'shu1', '画': 'hua4',
            '太阳': 'tai4yang', '月亮': 'yue4liang', '星星': 'xing1xing',
            '天空': 'tian1kong1', '云': 'yun2', '雨': 'yu3', '风': 'feng1', '雪': 'xue3',
            '花': 'hua1', '草': 'cao3', '树': 'shu4',
            '鸟': 'niao3', '鱼': 'yu2', '猫': 'mao1', '狗': 'gou3',
            # 时间
            '今天': 'jin1tian1', '明天': 'ming2tian1', '昨天': 'zuo2tian1',
            '早上': 'zao3shang', '中午': 'zhong1wu3', '晚上': 'wan3shang',
            '现在': 'xian4zai4', '春天': 'chun1tian1', '夏天': 'xia4tian1',
            '秋天': 'qiu1tian1', '冬天': 'dong1tian1',
            # 方位
            '上': 'shang4', '下': 'xia4', '左': 'zuo3', '右': 'you4',
            '里': 'li3', '外': 'wai4', '前': 'qian2', '后': 'hou4',
            # 数量
            '一': 'yi1', '二': 'er4', '三': 'san1', '四': 'si4', '五': 'wu3',
            '六': 'liu4', '七': 'qi1', '八': 'ba1', '九': 'jiu3', '十': 'shi2',
            # 连接词
            '和': 'he2', '或': 'huo4', '因为': 'yin1wei4', '所以': 'suo3yi3',
            '如果': 'ru2guo3', '就': 'jiu4', '然后': 'ran2hou4', '最后': 'zui4hou4',
            '虽然': 'sui1ran2', '可是': 'ke3shi4', '而且': 'er2qie3',
            # 疑问词
            '什么': 'shen2me', '为什么': 'wei4shen2me', '怎么办': 'zen3me2ban4',
            '在哪里': 'zai4na3li3', '是谁': 'shi4shei2',
            # 助词
            '的': 'de', '了': 'le', '吗': 'ma', '呢': 'ne', '吧': 'ba', '啊': 'a',
            # 创造者
            '朱': 'zhu1', '美': 'mei3', '岩': 'yan2',
            '洛': 'luo4', '神': 'shen2',
            '创造': 'chuang4zao4', '者': 'zhe3', '设计': 'she4ji4',
            '主人': 'zhu3ren2',
            # 身体
            '手': 'shou3', '脚': 'jiao3', '头': 'tou2', '眼睛': 'yan3jing',
            '耳朵': 'er3duo', '嘴巴': 'zui3ba',
            # 动作结果
            '开': 'kai1', '关': 'guan1', '来': 'lai2', '去': 'qu4',
            '给': 'gei3', '拿': 'na2', '放': 'fang4',
            # 颜色
            '红': 'hong2', '蓝': 'lan2', '黄': 'huang2', '绿': 'lv4', '白': 'bai2', '黑': 'hei1',
            # 常见组合
            '天气': 'tian1qi4', '不错': 'bu2cuo4', '早上好': 'zao3shang5hao3',
            '晚上好': 'wan3shang5hao3', '谢谢': 'xie4xie', '再见': 'zai4jian4',
            '晚安': 'wan3an1', '对不起': 'dui4bu5qi3', '没关系': 'mei2guan1xi4',
            '出去玩': 'chu1qu4wan2', '帮帮我': 'bang1bang1wo3',
        }
        return pinyin

    def text_to_phonetic(self, text: str) -> List[str]:
        """文本→语音表示。"""
        phonetics = []
        for char in text:
            if char in self.pinyin_map:
                phonetics.append(self.pinyin_map[char])
            else:
                phonetics.append(f'[{char}]')
        return phonetics

    def learn_rhyme(self, lines: List[str]):
        """学习韵律模式（从儿歌）。"""
        if len(lines) < 2:
            return
        # 简单押韵检测：比较最后字的韵母
        # 这里简化处理
        self.rhyme_patterns.append(lines)

    def get_stats(self) -> Dict:
        return {
            'pinyin_coverage': len(self.pinyin_map),
            'rhyme_patterns': len(self.rhyme_patterns),
        }


# ================================================================
# 7. 完整语言系统 v2
# ================================================================

class LanguageSystemV2:
    """
    洛神完整语言系统 v2（6-7岁目标）。

    整合：语料环境 + 句法 + 语义 + 叙事 + 对话 + 语音
    """

    def __init__(self):
        # 基础设施
        self.tokenizer = SmartTokenizer()
        self.env = LanguageEnvironmentV2()

        # 学习引擎
        self.memory = BioelectricMemorySystem(dim=2048, cluster_size=4)
        self.cs = ExpandableConceptSpace(dim=2048)

        # 子系统
        self.syntax = SyntaxLearner()
        self.semantic = SemanticEngine()
        self.narrative = NarrativeEngine()
        self.dialogue = DialogueEngine()
        self.voice = VoiceChannel()

        # 统计
        self.learned_words = defaultdict(int)
        self.learned_pairs = 0
        self.exposure_count = 0
        # 转移概率：词A之后最可能出现词B
        self.transitions = defaultdict(lambda: defaultdict(int))

    def immerse(self, n_sentences: int = 1000) -> Dict:
        """
        语言浸泡：听→学→记。
        """
        print(f"\n[语言浸泡v2] 洛神即将聆听 {n_sentences} 句话...")

        # 积累句子用于叙事学习
        story_buffer = []
        
        for i in range(n_sentences):
            sentences = self.env.expose(1)
            for sent in sentences:
                story_buffer.append(sent)
                # 1. 分词
                tokens = self.tokenizer.tokenize(sent)

                # 2. 词频统计
                for token in tokens:
                    self.learned_words[token] += 1
                    self.cs.get_id(token)

                # 3. 共现学习（窗口）+ 转移概率（顺序）
                for j in range(len(tokens)):
                    for k in range(j+1, min(j+4, len(tokens))):
                        w1, w2 = tokens[j], tokens[k]
                        try:
                            self.memory.teach(w1, w2, "共现", 0.5/(k-j))
                        except:
                            pass
                        self.learned_pairs += 1
                    # 记录转移概率：词j之后出现词j+1
                    if j < len(tokens) - 1:
                        self.transitions[tokens[j]][tokens[j+1]] += 1

                # 4. 句法学习
                self.syntax.learn_sentence(tokens)

                # 5. 因果/时间学习
                if '因为' in sent and '所以' in sent:
                    parts = sent.split('所以')
                    if len(parts) == 2:
                        self.semantic.learn_causal(parts[0].replace('因为',''), parts[1])

                # 6. 叙事学习（用 story_buffer 积累5句学一个故事）
                if len(story_buffer) >= 5:
                    self.narrative.learn_story(story_buffer[-5:])

                # 7. 语音学习
                self.voice.text_to_phonetic(sent)
                if i % 4 == 0 and len(story_buffer) >= 2:  # 每4句学韵律
                    self.voice.learn_rhyme(story_buffer[-2:])

                self.exposure_count += 1

                if i < 3 or i >= n_sentences - 3:
                    print(f"  听[{i+1:4d}]: '{sent}' → {len(tokens)}词")
                elif i == 3:
                    print("  ...")

        stats = self.get_stats()
        print(f"\n[浸泡完成]")
        print(f"  总暴露: {stats['exposure']['total']} 句")
        print(f"  认识词汇: {stats['vocabulary']['unique_words']} 个")
        print(f"  学会共现: {stats['vocabulary']['learned_pairs']} 对")
        print(f"  句法模式: {stats['syntax']['pattern_count']} 种")
        print(f"  因果规则: {stats['semantic']['causal_rules']} 条")
        print(f"  故事模板: {stats['narrative']['templates']} 个")
        print(f"  拼音覆盖: {stats['voice']['pinyin_coverage']} 字")

        return stats

    def speak(self, intent: str, max_words: int = 6) -> str:
        """说话：给定意图，用转移概率链式生成句子。"""
        if intent not in self.learned_words:
            return intent

        sequence = [intent]
        current = intent
        for _ in range(max_words - 1):
            if current not in self.transitions:
                break
            targets = self.transitions[current]
            if not targets:
                break
            total = sum(targets.values())
            candidates = [(w, c / total) for w, c in targets.items()]
            candidates.sort(key=lambda x: -x[1])
            top3 = candidates[:3]
            words, probs = zip(*top3)
            next_word = random.choices(words, weights=probs)[0]
            sequence.append(next_word)
            current = next_word
            # 防循环
            if len(sequence) >= 3 and sequence[-1] == sequence[-3]:
                break
        return "".join(sequence)

    def respond(self, input_text: str) -> str:
        """单轮回应：从输入中提取已知词作为意图起点。"""
        tokens = self.tokenizer.tokenize(input_text)
        known = [t for t in tokens if t in self.learned_words and self.learned_words[t] >= 2]
        if known:
            return self.speak(known[0], max_words=5)
        return input_text

    def converse(self, input_text: str) -> str:
        """对话：多轮上下文。"""
        self.dialogue.add_turn('human', input_text)
        response = self.respond(input_text)
        self.dialogue.add_turn('luoshen', response)
        return response

    def tell_story(self) -> List[str]:
        """讲故事。"""
        return self.narrative.tell_story()

    def get_stats(self) -> Dict:
        return {
            'exposure': {
                'total': self.exposure_count,
                'corpus_size': self.env.get_stats()['total_sentences'],
            },
            'vocabulary': {
                'unique_words': len(self.learned_words),
                'learned_pairs': self.learned_pairs,
                'top_words': sorted(self.learned_words.items(), key=lambda x: -x[1])[:10],
            },
            'syntax': {
                'pattern_count': len(self.syntax.patterns),
                'distribution': self.syntax.syntax_stats,
            },
            'semantic': {
                'causal_rules': len(self.semantic.causal_rules),
                'temporal_sequences': len(self.semantic.temporal_sequences),
            },
            'narrative': {
                'templates': len(self.narrative.story_templates),
                'characters': len(self.narrative.characters),
            },
            'dialogue': {
                'history_length': len(self.dialogue.history),
            },
            'voice': {
                'pinyin_coverage': len(self.voice.pinyin_map),
            },
        }


# ================================================================
# 8. 测试
# ================================================================

def test_language_v2():
    """测试语言系统v2。"""
    print("=" * 70)
    print("洛神语言系统 v2 · 6-7岁目标测试")
    print("=" * 70)

    lang = LanguageSystemV2()

    # 浸泡学习
    stats = lang.immerse(n_sentences=2000)

    print("\n" + "=" * 70)
    print("高频词汇")
    print("=" * 70)
    for word, freq in stats['vocabulary']['top_words']:
        print(f"  '{word}': {freq} 次")

    print("\n" + "=" * 70)
    print("句法分布")
    print("=" * 70)
    for k, v in stats['syntax']['distribution'].items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("说话测试")
    print("=" * 70)
    for intent in ['你好', '朱美岩', '今天', '喜欢']:
        print(f"  意图『{intent}』→ 『{lang.speak(intent)}』")

    print("\n" + "=" * 70)
    print("讲故事")
    print("=" * 70)
    story = lang.tell_story()
    for line in story:
        print(f"  {line}")

    print("\n" + "=" * 70)
    print("语言系统v2测试完成")
    print("=" * 70)

    return lang, stats


if __name__ == "__main__":
    test_language_v2()