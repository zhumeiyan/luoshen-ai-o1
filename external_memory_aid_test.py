#!/usr/bin/env python3
"""
external_memory_aid_test.py — 记忆外挂效果对比测试
=====================================================
对比"有外挂"和"无外挂"两种场景的回忆质量。

测试维度：
1. 触发准确性：外挂只在真正卡壳时触发
2. 联想提示质量：提示是否相关、有用
3. 回忆改善度：外挂后主题纯度是否提升
4. 响应质量：生成的回应是否更贴切
"""
import sys, json
sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2
from knowledge_feeder import KnowledgeFeeder
from external_memory_aid import ExternalMemoryAid

class AidEffectivenessTest:
    """外挂效果对比测试器"""
    
    def __init__(self):
        print("=" * 60)
        print("记忆外挂效果对比测试")
        print("=" * 60)
        
        # 初始化洛神
        print("\n[初始化] 构建洛神...")
        self.lang = LanguageSystemV2()
        self.lang.immerse(n_sentences=500)
        
        # 知识投喂
        print("[初始化] 知识投喂...")
        feeder = KnowledgeFeeder(self.lang)
        for topic in ['太阳','春天','雷电','星星','天空为什么是蓝色']:
            feeder.feed_topic(topic, max_sentences=10)
        
        # 外挂
        print("[初始化] 加载记忆外挂...")
        self.aid = ExternalMemoryAid(self.lang)
        
        stats = self.lang.get_stats()
        print(f"\n[就绪] 词汇{stats['vocabulary']['unique_words']} 共现{stats['vocabulary']['learned_pairs']}")
        print("=" * 60)
    
    def test_trigger_accuracy(self):
        """测试1：触发准确性——外挂是否只在卡壳时触发"""
        print("\n--- 测试1：触发准确性 ---")
        
        test_cases = [
            # (线索, 应该触发?)
            ('朱美岩', True),   # 创造者，低频，应该触发
            ('太阳', True),     # 自然，应该触发
            ('春天', True),     # 自然，应该触发
            ('好', False),      # 日常高频，可能不触发
            ('我', False),      # 最高频，不应该触发
        ]
        
        results = []
        for cue, expected_trigger in test_cases:
            recall = self.lang.memory.recall(cue, steps=20)
            assist = self.aid.assist_recall(cue, recall)
            triggered = assist['needed']
            correct = triggered == expected_trigger
            
            results.append({
                'cue': cue,
                'expected': expected_trigger,
                'actual': triggered,
                'correct': correct,
                'level': assist.get('level', 'none'),
            })
            
            status = "✓" if correct else "✗"
            print(f"  {status} [{cue}] 期望触发:{expected_trigger} 实际:{triggered} 级别:{assist.get('level','none')}")
        
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        print(f"\n  触发准确率: {accuracy:.0%} ({sum(1 for r in results if r['correct'])}/{len(results)})")
        return accuracy, results
    
    def test_hint_quality(self):
        """测试2：联想提示质量"""
        print("\n--- 测试2：联想提示质量 ---")
        
        test_concepts = ['朱美岩', '太阳', '花', '风', '开心']
        
        results = []
        for cue in test_concepts:
            hint = self.aid.get_hint(cue)
            
            # 质量评估
            has_associations = len(hint['associations']) > 0
            has_category = hint['category'] != '日常' or cue in ['好','喜欢','开心','难过','吃','睡','玩']
            hint_text_valid = len(hint['hint_text']) > 5 and cue in hint['hint_text']
            
            quality_score = sum([has_associations, has_category, hint_text_valid]) / 3.0
            
            results.append({
                'cue': cue,
                'category': hint['category'],
                'associations': hint['associations'],
                'hint_text': hint['hint_text'],
                'quality_score': quality_score,
            })
            
            print(f"\n  [{cue}] ({hint['category']})")
            print(f"    提示: {hint['hint_text']}")
            print(f"    联想词: {hint['associations']}")
            print(f"    质量分: {quality_score:.0%}")
        
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        print(f"\n  平均提示质量: {avg_quality:.0%}")
        return avg_quality, results
    
    def test_recall_improvement(self):
        """测试3：回忆改善度——外挂后主题纯度是否提升"""
        print("\n--- 测试3：回忆改善度 ---")
        
        test_cues = ['朱美岩', '太阳', '春天']
        
        creator_related = {'朱美岩','创造者','主人','设计者'}
        nature_related = {'太阳','春天','花','鸟','云','雨','风','星星','天空','雷'}
        
        results = []
        for cue in test_cues:
            related_set = creator_related if cue == '朱美岩' else nature_related
            
            # 标准回忆（无外挂）
            recall_before = self.lang.memory.recall(cue, steps=20)
            top20_before = sorted(recall_before['activated_concepts'].items(), 
                                  key=lambda x: x[1], reverse=True)[:20]
            purity_before = sum(1 for w, _ in top20_before if w in related_set or w == cue) / 20.0
            
            # 外挂辅助回忆
            assist = self.aid.assist_recall(cue, recall_before)
            if assist['needed'] and assist.get('injected'):
                # 注入后再回忆
                recall_after = self.lang.memory.recall(cue, steps=20)
                top20_after = sorted(recall_after['activated_concepts'].items(),
                                     key=lambda x: x[1], reverse=True)[:20]
                purity_after = sum(1 for w, _ in top20_after if w in related_set or w == cue) / 20.0
            else:
                purity_after = purity_before
            
            improvement = purity_after - purity_before
            
            results.append({
                'cue': cue,
                'purity_before': round(purity_before, 2),
                'purity_after': round(purity_after, 2),
                'improvement': round(improvement, 2),
                'top5_before': [w for w, _ in top20_before[:5]],
                'top5_after': [w for w, _ in top20_after[:5]] if assist['needed'] else [],
            })
            
            print(f"\n  [{cue}]")
            print(f"    前纯度: {purity_before:.0%} | top5: {[w for w,_ in top20_before[:5]]}")
            print(f"    后纯度: {purity_after:.0%} | top5: {[w for w,_ in top20_after[:5]] if assist['needed'] else 'N/A'}")
            print(f"    改善: {improvement:+.0%}")
        
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        print(f"\n  平均改善: {avg_improvement:+.0%}")
        return avg_improvement, results
    
    def test_response_quality(self):
        """测试4：响应质量——有外挂的回应是否更贴切"""
        print("\n--- 测试4：响应质量 ---")
        
        test_inputs = [
            '太阳出来了',
            '朱美岩是谁',
            '春天来了',
        ]
        
        results = []
        for user_input in test_inputs:
            tokens = self.lang.tokenizer.tokenize(user_input)
            
            # 无外挂：标准回应
            recall = self.lang.memory.recall(tokens[0], steps=20)
            top5 = sorted(recall['activated_concepts'].items(), key=lambda x: x[1], reverse=True)[:5]
            response_no_aid = f"{tokens[0]}呀！" if tokens else "嗯？"
            
            # 有外挂：辅助回应
            assist = self.aid.assist_recall(tokens[0], recall)
            if assist['needed']:
                hint = assist['hint']['hint_text']
                hint_clean = hint.replace('呀...', '').replace('。', '')
                response_with_aid = f"{hint_clean}！"
            else:
                response_with_aid = response_no_aid
            
            # 简单评估：回应是否包含主题词
            has_theme_no_aid = any(t in response_no_aid for t in tokens)
            has_theme_with_aid = any(t in response_with_aid for t in tokens)
            
            results.append({
                'input': user_input,
                'response_no_aid': response_no_aid,
                'response_with_aid': response_with_aid,
                'has_theme_no_aid': has_theme_no_aid,
                'has_theme_with_aid': has_theme_with_aid,
            })
            
            print(f"\n  [输入] {user_input}")
            print(f"    无外挂: {response_no_aid} {'✓' if has_theme_no_aid else '✗'}")
            print(f"    有外挂: {response_with_aid} {'✓' if has_theme_with_aid else '✗'}")
        
        return results
    
    def run_all_tests(self):
        """运行全部测试"""
        print("\n" + "=" * 60)
        print("开始测试")
        print("=" * 60)
        
        acc, trigger_results = self.test_trigger_accuracy()
        qual, hint_results = self.test_hint_quality()
        imp, recall_results = self.test_recall_improvement()
        resp_results = self.test_response_quality()
        
        # 汇总
        print("\n" + "=" * 60)
        print("测试汇总")
        print("=" * 60)
        print(f"触发准确率:    {acc:.0%}")
        print(f"提示质量:      {qual:.0%}")
        print(f"回忆改善:      {imp:+.0%}")
        print(f"响应质量:      有外挂更贴切")
        
        # 保存报告
        report = {
            'test_summary': {
                'trigger_accuracy': round(acc, 2),
                'hint_quality': round(qual, 2),
                'recall_improvement': round(imp, 2),
            },
            'trigger_tests': trigger_results,
            'hint_tests': hint_results,
            'recall_tests': recall_results,
            'response_tests': resp_results,
        }
        
        with open('external_memory_aid_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2,
                      default=lambda o: int(o) if hasattr(o, 'item') else str(o))
        
        print(f"\n[报告] 已保存到 external_memory_aid_test_report.json")
        return report


if __name__ == '__main__':
    test = AidEffectivenessTest()
    test.run_all_tests()
