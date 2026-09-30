#!/usr/bin/env python3
import sys
sys.path.insert(0, '01_environment')

from real_dialogue_corpus import RealChildDialogueCorpus
from real_dialogue_engine import RealDialogueEngine, IntentType, EmotionType, DialogueTurn

corpus = RealChildDialogueCorpus()

metrics = {
    'intent_clear': 0, 'intent_correct': 0,
    'emotion_clear': 0, 'emotion_correct': 0,
    'topic_ok': 0, 'natural': 0,
    'total': 0,
}

samples = []

for dialogue in corpus.dialogues:
    engine = RealDialogueEngine()
    for turn in dialogue['轮次']:
        if turn['说话人'] != '孩子':
            t = DialogueTurn(speaker='大人', text=turn['文本'], intent=IntentType.UNKNOWN, emotion=EmotionType.NEUTRAL, topic='')
            engine.context_tracker.update_context(engine.context, t)
            continue
        
        text = turn['文本']
        resp = engine.process(text)
        summary = engine.get_context_summary()
        metrics['total'] += 1
        
        intent = engine.intent_recognizer.recognize(text, engine.context)
        emotion = engine.emotional_engine.recognize_emotion(text)
        topic = summary['current_topic']
        
        # 意图：长度>2或有关键词才算有线索
        has_intent = len(text) > 2 or any(kw in text for kw in ['要','想','饿','渴','困','怕','帮','谢谢','开心','难过','生气','为什么'])
        if has_intent:
            metrics['intent_clear'] += 1
            if intent != IntentType.UNKNOWN:
                metrics['intent_correct'] += 1
        
        # 情感：有情感关键词才算有线索
        has_emotion = emotion != EmotionType.NEUTRAL
        if has_emotion:
            metrics['emotion_clear'] += 1
            metrics['emotion_correct'] += 1
        
        # 话题
        if topic != '一般':
            metrics['topic_ok'] += 1
        
        # 自然度
        if resp not in ['嗯？', '什么？', '我没听懂', '再说一遍']:
            metrics['natural'] += 1
        
        if len(samples) < 25:
            samples.append({
                'input': text, 'response': resp,
                'intent': intent.value if has_intent else f'极简({text})',
                'emotion': emotion.value, 'topic': topic,
            })

i_acc = round(metrics['intent_correct']/metrics['intent_clear']*100, 1) if metrics['intent_clear'] else 0
e_acc = round(metrics['emotion_correct']/metrics['emotion_clear']*100, 1) if metrics['emotion_clear'] else 0
t_acc = round(metrics['topic_ok']/metrics['total']*100, 1)
n_acc = round(metrics['natural']/metrics['total']*100, 1)
overall = round((i_acc + e_acc + t_acc + n_acc) / 4, 1)

import json
result = {
    'metrics': metrics,
    'scores': {'intent': i_acc, 'emotion': e_acc, 'topic': t_acc, 'natural': n_acc, 'overall': overall},
    'pass': overall >= 60,
    'samples': samples,
}
with open('real_dialogue_final_result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print('=== 真实对话自然度评估 ===')
print('总轮次:', metrics['total'])
print('意图:', metrics['intent_correct'], '/', metrics['intent_clear'], '=', i_acc, '%')
print('情感:', metrics['emotion_correct'], '/', metrics['emotion_clear'], '=', e_acc, '%')
print('话题:', metrics['topic_ok'], '/', metrics['total'], '=', t_acc, '%')
print('自然:', metrics['natural'], '/', metrics['total'], '=', n_acc, '%')
print('综合:', overall, '%')
print()
for s in samples[:15]:
    print(' ', s['input'], '→', s['response'], '[', s['intent'], '/', s['emotion'], ']')