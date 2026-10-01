#!/usr/bin/env python3
"""
voice_rhythm_engine.py — 语音节奏引擎
======================================

为句子添加TTS（文字转语音）友好的停顿和语调标记。

核心原理：
- 标点决定大停顿（。！？）
- 语法结构决定小停顿（主语后、并列处）
- 情感决定语调（开心=升调，难过=降调）
- 逻辑关系决定停顿长短（因果>并列）

标记系统：
- [P:long] 长停顿（1秒）
- [P:short] 短停顿（0.3秒）
- [T:up] 升调（疑问/开心）
- [T:down] 降调（陈述/难过）
- [T:flat] 平调（平静）
- [S:slow] 慢速（强调）
- [S:fast] 快速（兴奋）

不是后处理，是生成时就嵌入节奏信息。
"""

import re
from typing import Dict, List, Tuple


class VoiceRhythmEngine:
    """
    语音节奏引擎。
    
    给句子注入生命：停顿、语调、速度。
    """
    
    def __init__(self):
        # 情感→语调映射
        self.emotion_tones = {
            '开心': {'base': 'up', 'end': 'up', 'speed': 'normal'},
            '难过': {'base': 'down', 'end': 'down', 'speed': 'slow'},
            '害怕': {'base': 'up', 'end': 'up', 'speed': 'fast'},
            '愤怒': {'base': 'down', 'end': 'down', 'speed': 'fast'},
            '平静': {'base': 'flat', 'end': 'flat', 'speed': 'normal'},
            '好奇': {'base': 'up', 'end': 'up', 'speed': 'normal'},
            '惊讶': {'base': 'up', 'end': 'up', 'speed': 'fast'},
        }
        
        # 句法模式→停顿映射
        self.syntax_pauses = {
            '因为': ('before', 'long'),    # 原因状语前长停顿
            '所以': ('before', 'medium'),  # 结果前中等停顿
            '如果': ('before', 'long'),    # 条件前长停顿
            '虽然': ('before', 'long'),    # 转折前长停顿
            '但是': ('before', 'medium'),  # 转折后中等停顿
            '而且': ('before', 'short'),   # 并列前短停顿
            '也': ('before', 'short'),     # 并列前短停顿
            '先': ('after', 'short'),      # 顺序标记后短停
            '再': ('after', 'short'),      # 顺序标记后短停
            '最后': ('before', 'medium'),  # 最后前中等停顿
        }
    
    def mark_rhythm(self, sentence: str, emotion: str = '平静') -> str:
        """
        为句子添加节奏标记。
        
        输入: "因为下雨所以地湿"
        输出: "[P:long]因为下雨[P:medium]所以地湿。"
        """
        marked = sentence
        
        # 1. 句法停顿标记
        for marker, (position, pause_type) in self.syntax_pauses.items():
            if marker in marked:
                if position == 'before':
                    marked = marked.replace(marker, f'[P:{pause_type}]{marker}', 1)
                else:
                    # after: 在标记后加停顿
                    idx = marked.find(marker)
                    if idx >= 0:
                        end_idx = idx + len(marker)
                        marked = marked[:end_idx] + f'[P:{pause_type}]' + marked[end_idx:]
        
        # 2. 句尾语调标记
        tone = self.emotion_tones.get(emotion, {'end': 'flat'})
        
        # 根据句尾标点决定语调
        if marked.endswith('？') or marked.endswith('?'):
            marked += f'[T:{tone["end"]}]'
        elif marked.endswith('！') or marked.endswith('!'):
            marked += f'[T:{tone["end"]}][S:fast]'
        elif marked.endswith('。') or marked.endswith('.'):
            marked += f'[T:{tone["end"]}][P:long]'
        elif marked.endswith('...'):
            marked += f'[T:{tone["end"]}][P:long][S:slow]'
        else:
            # 没有标点，添加句号和语调
            marked += f'。[T:{tone["end"]}][P:long]'
        
        # 3. 句子开头的语调基调
        marked = f'[T:{tone["base"]}]{marked}'
        
        # 4. 全局速度标记
        if tone['speed'] != 'normal':
            marked = f'[S:{tone["speed"]}]{marked}'
        
        return marked
    
    def mark_monologue(self, sentences: List[str], emotion: str = '平静') -> List[str]:
        """
        为整段独白添加节奏标记。
        
        句间停顿：根据逻辑关系决定。
        """
        marked_sentences = []
        
        for i, sent in enumerate(sentences):
            marked = self.mark_rhythm(sent, emotion)
            
            # 句间停顿：如果不是最后一句，添加句间停顿
            if i < len(sentences) - 1:
                # 检查下一句的开头，判断逻辑关系
                if i + 1 < len(sentences):
                    next_sent = sentences[i + 1]
                    if any(m in next_sent for m in ['因为', '如果', '虽然']):
                        marked += '[P:long]'  # 逻辑连接前长停
                    elif any(m in next_sent for m in ['所以', '但是', '而且']):
                        marked += '[P:medium]'  # 承接前中等停
                    else:
                        marked += '[P:short]'  # 普通短停
            
            marked_sentences.append(marked)
        
        return marked_sentences
    
    def to_tts_friendly(self, marked_sentence: str) -> str:
        """
        转换为TTS友好的纯文本（带SSML标记）。
        
        把自定义标记转成标准SSML：
        [P:long] → <break time="1000ms"/>
        [T:up] → <prosody pitch="+20%">
        """
        ssml = marked_sentence
        
        # 停顿转换
        ssml = ssml.replace('[P:long]', '<break time="1000ms"/>')
        ssml = ssml.replace('[P:medium]', '<break time="600ms"/>')
        ssml = ssml.replace('[P:short]', '<break time="300ms"/>')
        
        # 语调转换
        ssml = ssml.replace('[T:up]', '<prosody pitch="+20%">')
        ssml = ssml.replace('[T:down]', '<prosody pitch="-20%">')
        ssml = ssml.replace('[T:flat]', '<prosody pitch="default">')
        
        # 速度转换
        ssml = ssml.replace('[S:slow]', '<prosody rate="80%">')
        ssml = ssml.replace('[S:fast]', '<prosody rate="120%">')
        ssml = ssml.replace('[S:normal]', '<prosody rate="100%">')
        
        # 关闭未关闭的prosody标签
        # 简化处理：在句尾关闭
        if '<prosody' in ssml and '</prosody>' not in ssml:
            ssml += '</prosody>'
        
        return f'<speak>{ssml}</speak>'


def test_voice_rhythm():
    """测试语音节奏引擎。"""
    print("=" * 60)
    print("语音节奏引擎测试")
    print("=" * 60)
    
    engine = VoiceRhythmEngine()
    
    # 测试句子
    test_cases = [
        ("因为下雨，所以地湿", "平静"),
        ("太阳出来了！", "开心"),
        ("如果努力，就成功", "平静"),
        ("虽然很难，但是要坚持", "难过"),
        ("花让我开心", "开心"),
    ]
    
    print("\n节奏标记：")
    for sent, emotion in test_cases:
        marked = engine.mark_rhythm(sent, emotion)
        print(f"\n[{emotion}] {sent}")
        print(f"  标记: {marked}")
        
        # 转SSML
        ssml = engine.to_tts_friendly(marked)
        print(f"  SSML: {ssml}")
    
    # 测试整段独白
    print("\n" + "=" * 60)
    print("整段独白节奏：")
    monologue = [
        "太阳很好",
        "因为天气暖",
        "所以花开了",
        "我很开心",
    ]
    marked_mono = engine.mark_monologue(monologue, emotion='开心')
    for sent in marked_mono:
        print(f"  {sent}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_voice_rhythm()
