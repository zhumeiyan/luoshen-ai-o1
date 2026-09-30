#!/usr/bin/env python3
"""
main_with_aid.py — 带记忆外挂的洛神对话演示
============================================
演示 ExternalMemoryAid 如何集成到洛神的对话流中。

核心流程：
1. 用户说一句话
2. 洛神尝试用生物电记忆理解/回应
3. 如果回忆卡壳（主题纯度低），自动调用外挂层
4. 外挂提供联想提示或查知识库
5. 将外挂结果临时注入记忆，辅助回应生成

这不是替代 main.py，而是展示集成方式。
"""
import sys, json, os
from typing import Dict, List
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_environment'))

from language_learner_v2 import LanguageSystemV2
from knowledge_feeder import KnowledgeFeeder
from external_memory_aid import ExternalMemoryAid

class LuoshenWithAid:
    """
    带记忆外挂的洛神。
    
    对话时自动检测记忆卡壳，调用外挂辅助。
    """
    
    def __init__(self):
        print("=" * 60)
        print("洛神 · 带记忆外挂版")
        print("=" * 60)
        
        # 1. 语言系统（生物电记忆）
        print("\n[初始化] 语言系统...")
        self.lang = LanguageSystemV2()
        self.lang.immerse(n_sentences=500)
        
        # 2. 知识投喂
        print("[初始化] 知识投喂...")
        feeder = KnowledgeFeeder(self.lang)
        for topic in ['太阳','春天','雷电','星星','天空为什么是蓝色']:
            feeder.feed_topic(topic, max_sentences=10)
        
        # 3. 记忆外挂（工具层）
        print("[初始化] 记忆外挂...")
        self.aid = ExternalMemoryAid(self.lang)
        
        # 4. 统计
        stats = self.lang.get_stats()
        print(f"\n[就绪] 洛神已准备好对话")
        print(f"  词汇: {stats['vocabulary']['unique_words']} 个")
        print(f"  共现: {stats['vocabulary']['learned_pairs']} 对")
        print(f"  句法: {stats['syntax']['pattern_count']} 种")
        print("=" * 60)
    
    def chat(self, user_input: str) -> Dict:
        """
        对话入口：理解用户输入 → 回忆相关知识 → 生成回应。
        
        关键创新：回忆时自动检测卡壳，调用外挂辅助。
        """
        print(f"\n[用户] {user_input}")
        
        # 1. 分词，提取关键概念
        tokens = self.lang.tokenizer.tokenize(user_input)
        print(f"[理解] 分词: {tokens}")
        
        # 2. 对每个关键概念尝试回忆
        recall_results = {}
        aid_activations = []
        
        # 过滤虚词：只检查实词（名词、动词、形容词）
        stop_words = {'是','的','了','在','吗','不','我','你','他','她','它',
                      '我们','你们','他们','什么','怎么','为什么','谁','哪',
                      '这','那','个','些','和','与','或','但','如果','因为',
                      '所以','然后','就','都','也','很','非常','特别','太',
                      '是谁','为什么','怎么'}  # 常见疑问词组
        
        for token in tokens:
            if len(token) < 2 or token in stop_words:
                continue
            
            # 生物电回忆
            recall = self.lang.memory.recall(token, steps=20)
            recall_results[token] = recall
            
            # 检测是否需要外挂
            assist = self.aid.assist_recall(token, recall)
            
            if assist['needed']:
                print(f"  [外挂触发] '{token}' 卡壳 → 级别:{assist['level']}")
                print(f"    提示: {assist['hint']['hint_text']}")
                
                if assist.get('knowledge'):
                    print(f"    知识库: 查到{len(assist['knowledge']['explanations'])}条解释")
                
                aid_activations.append({
                    'token': token,
                    'level': assist['level'],
                    'hint': assist['hint']['hint_text'],
                })
        
        # 3. 生成回应（简化版：基于回忆结果生成句子）
        response = self._generate_response(tokens, recall_results, aid_activations)
        
        print(f"[洛神] {response}")
        
        return {
            'user_input': user_input,
            'tokens': tokens,
            'recall_results': {t: len(r['activated_concepts']) for t, r in recall_results.items()},
            'aid_activations': aid_activations,
            'response': response,
        }
    
    def _generate_response(self, tokens, recall_results, aid_activations) -> str:
        """
        基于回忆结果生成回应。
        
        如果有外挂激活，在回应中体现辅助效果。
        """
        # 简单策略：检查tokens中是否有已知概念，组合成句
        known_words = set(self.lang.learned_words.keys())
        matched = [t for t in tokens if t in known_words]
        
        if not matched:
            return "嗯...你在说什么呀？"
        
        # 如果有外挂激活，用提示词丰富回应
        if aid_activations:
            # 取第一个外挂的提示
            hint = aid_activations[0]['hint']
            # 简化提示，去掉"呀..."等语气词
            hint_clean = hint.replace('呀...', '').replace('。', '')
            
            # 生成带提示的回应
            responses = [
                f"{hint_clean}！",
                f"我知道{hint_clean}！",
                f"{hint_clean}，对吗？",
            ]
            import random
            return random.choice(responses)
        
        # 无外挂，标准回应
        if len(matched) >= 2:
            return f"{'和'.join(matched[:2])}呀！"
        else:
            return f"{matched[0]}！"
    
    def demo_conversation(self):
        """演示对话：展示外挂如何在不同场景触发。"""
        print("\n" + "=" * 60)
        print("对话演示：记忆外挂自动触发")
        print("=" * 60)
        
        test_inputs = [
            "太阳出来了",           # 自然概念，应能正常回忆
            "朱美岩是谁",           # 创造者概念，可能卡壳，触发外挂
            "天空为什么是蓝色",     # 复杂概念，可能卡壳
            "春天来了",             # 自然概念
            "我喜欢玩",             # 日常概念
        ]
        
        results = []
        for user_input in test_inputs:
            result = self.chat(user_input)
            results.append(result)
            print()
        
        # 统计
        n_aid = sum(len(r['aid_activations']) for r in results)
        print("=" * 60)
        print("演示总结")
        print("=" * 60)
        print(f"总对话: {len(results)} 轮")
        print(f"外挂触发: {n_aid} 次")
        print(f"触发率: {n_aid/len(results):.0%}")
        
        for r in results:
            if r['aid_activations']:
                print(f"\n  [{r['user_input']}]")
                for a in r['aid_activations']:
                    print(f"    → '{a['token']}' 级别:{a['level']}")
        
        return results


def main():
    """主程序：运行带外挂的洛神对话演示。"""
    luoshen = LuoshenWithAid()
    results = luoshen.demo_conversation()
    
    # 保存结果
    report = {
        'description': '记忆外挂集成演示',
        'conversations': [
            {
                'input': r['user_input'],
                'tokens': r['tokens'],
                'aid_activations': r['aid_activations'],
                'response': r['response'],
            }
            for r in results
        ],
    }
    
    with open('luoshen_with_aid_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else str(o))
    
    print(f"\n[报告] 已保存到 luoshen_with_aid_report.json")


if __name__ == '__main__':
    main()
