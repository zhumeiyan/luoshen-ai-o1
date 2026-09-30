#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
language_gap_analysis.py — 6-7岁儿童语言能力差距分析
============================================================

对照人类儿童6-7岁语言发展里程碑，分析洛神当前水平与缺口。

数据来源：发展心理学语言发展阶段理论（Brown, 1973; Bloom & Lahey, 1978;
中国儿童语言发展常模，2020）

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
差距通过环境层补充来弥补。
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class LanguageMilestone:
    """语言发展里程碑项。"""
    name: str           # 能力名称
    age_target: str     # 目标年龄
    description: str    # 描述
    criteria: List[str] # 达标标准
    luoshen_current: str  # 洛神当前水平
    gap_level: str      # 差距等级：微小/中等/严重/致命
    environment_needed: str  # 需要补充的环境


class LanguageGapAnalyzer:
    """
    6-7岁儿童语言能力差距分析器。
    """
    
    def __init__(self):
        self.milestones = self._build_milestones()
        
    def _build_milestones(self) -> List[LanguageMilestone]:
        """构建10项语言能力里程碑。"""
        return [
            LanguageMilestone(
                name="词汇量",
                age_target="6-7岁",
                description="能理解并主动使用的词汇总量",
                criteria=[
                    "被动词汇3000-5000个（能听懂）",
                    "主动词汇2000-3000个（能使用）",
                    "掌握抽象词（时间、空间、情感）",
                    "能根据语境选择合适词汇",
                ],
                luoshen_current="355个2字碎片（'你好''我想'等），不是完整词汇，无语义层级",
                gap_level="严重",
                environment_needed="5000+句真实语料浸泡，包含抽象概念、同义词辨析、语境示例"
            ),
            
            LanguageMilestone(
                name="句法结构",
                age_target="6-7岁",
                description="使用复杂句法结构的能力",
                criteria=[
                    "主谓宾完整句",
                    "使用定语、状语、补语",
                    "复合句（因为...所以...、虽然...但是...）",
                    "疑问句、否定句、祈使句",
                    "能调整语序表达强调",
                ],
                luoshen_current="只有转移概率链式拼接（'我想'→'吃饭'），无句法规则，无层级结构",
                gap_level="致命",
                environment_needed="结构化语料（主谓宾标注句、复合句模板、句法变换练习）"
            ),
            
            LanguageMilestone(
                name="语义深度理解",
                age_target="6-7岁",
                description="理解语言背后的意义，而非字面",
                criteria=[
                    "理解因果关系（因为...所以...）",
                    "理解时间顺序（先...然后...最后...）",
                    "理解指代消解（他/她/它指谁）",
                    "理解隐喻和比喻",
                    "能推断言外之意",
                ],
                luoshen_current="纯统计共现，无语义层级，不理解'因为所以'的因果逻辑",
                gap_level="致命",
                environment_needed="因果故事语料、指代训练、逻辑推理场景、语义角色标注"
            ),
            
            LanguageMilestone(
                name="叙事能力",
                age_target="6-7岁",
                description="讲述完整故事的能力",
                criteria=[
                    "故事有开头、发展、高潮、结局",
                    "有角色设定和动机",
                    "使用时间连接词",
                    "能复述听过的故事",
                    "能创作简单故事",
                ],
                luoshen_current="无叙事结构，只能输出2-4个词的碎片链",
                gap_level="严重",
                environment_needed="绘本故事语料（100+个完整故事）、叙事模板、角色-情节-结局训练"
            ),
            
            LanguageMilestone(
                name="对话能力",
                age_target="6-7岁",
                description="维持多轮对话的能力",
                criteria=[
                    "能维持5轮以上对话",
                    "理解上下文（知道刚才在聊什么）",
                    "能进行话题转换",
                    "能提问和回答",
                    "能表达同意/不同意",
                ],
                luoshen_current="单轮回应，无对话历史记忆，无上下文维持",
                gap_level="严重",
                environment_needed="对话语料（多轮问答、话题转换、辩论场景）"
            ),
            
            LanguageMilestone(
                name="阅读准备",
                age_target="6-7岁",
                description="识字和理解书面语言的能力",
                criteria=[
                    "认识500-1000个常见字",
                    "能读简单绘本",
                    "理解标点符号",
                    "能根据上下文猜字义",
                ],
                luoshen_current="无字符识别能力，只有2字组合的统计关联",
                gap_level="中等",
                environment_needed="识字卡语料、绘本文本、字形-字义-字音三联学习"
            ),
            
            LanguageMilestone(
                name="书写准备",
                age_target="6-7岁",
                description="书写表达的能力",
                criteria=[
                    "能写简单句子",
                    "基本正确的标点使用",
                    "能写留言条/简单信件",
                    "拼写常见词",
                ],
                luoshen_current="无书写能力，输出是文本拼接",
                gap_level="中等",
                environment_needed="书写模板训练、留言条范例、拼写练习"
            ),
            
            LanguageMilestone(
                name="语音意识",
                age_target="6-7岁",
                description="对语音结构的感知和操作能力",
                criteria=[
                    "能分辨声母、韵母",
                    "掌握拼音规则",
                    "能分辨声调",
                    "能发现押韵",
                    "能进行音节拆分/组合",
                ],
                luoshen_current="无语音通道，只有文本符号学习",
                gap_level="严重",
                environment_needed="拼音语料、声调训练、儿歌韵律、音节拆分练习"
            ),
            
            LanguageMilestone(
                name="社会语言",
                age_target="6-7岁",
                description="根据场合调整语言的能力",
                criteria=[
                    "对长辈用礼貌语",
                    "对朋友用随意语",
                    "知道什么场合说什么话",
                    "能理解语言的社会含义",
                ],
                luoshen_current="无社会语境区分，所有输入统一处理",
                gap_level="中等",
                environment_needed="社会场景语料（家庭/学校/公共场合）、礼貌用语训练、角色扮演对话"
            ),
            
            LanguageMilestone(
                name="元语言意识",
                age_target="6-7岁",
                description="对语言本身进行反思的能力",
                criteria=[
                    "能讨论词的意思",
                    "能解释句子结构",
                    "能发现语言错误",
                    "能进行简单翻译",
                ],
                luoshen_current="无元认知能力，不能讨论自己的语言",
                gap_level="中等",
                environment_needed="元语言对话（'这个词什么意思'）、纠错训练、解释练习"
            ),
        ]
    
    def analyze(self) -> Dict:
        """执行完整差距分析。"""
        results = {
            'milestones': [],
            'summary': {
                'total': len(self.milestones),
                'fatal': 0,
                'severe': 0,
                'moderate': 0,
                'minor': 0,
            }
        }
        
        for m in self.milestones:
            results['milestones'].append({
                'name': m.name,
                'age_target': m.age_target,
                'description': m.description,
                'criteria': m.criteria,
                'luoshen_current': m.luoshen_current,
                'gap_level': m.gap_level,
                'environment_needed': m.environment_needed,
            })
            
            # 统计差距等级
            if m.gap_level == "致命":
                results['summary']['fatal'] += 1
            elif m.gap_level == "严重":
                results['summary']['severe'] += 1
            elif m.gap_level == "中等":
                results['summary']['moderate'] += 1
            else:
                results['summary']['minor'] += 1
        
        return results
    
    def get_priority_modules(self) -> List[Tuple[str, str, str]]:
        """获取按优先级排序的待补模块。"""
        priority = []
        for m in self.milestones:
            if m.gap_level in ["致命", "严重"]:
                priority.append((m.name, m.gap_level, m.environment_needed))
        # 致命优先
        priority.sort(key=lambda x: 0 if x[1] == "致命" else 1)
        return priority
    
    def print_report(self):
        """打印差距分析报告。"""
        results = self.analyze()
        
        print("=" * 70)
        print("洛神语言能力差距分析：6-7岁儿童标准")
        print("=" * 70)
        
        for i, m in enumerate(results['milestones'], 1):
            level_color = {"致命": "🔴", "严重": "🟠", "中等": "🟡", "微小": "🟢"}
            icon = level_color.get(m['gap_level'], "⚪")
            print(f"\n{i}. {icon} {m['name']} [{m['gap_level']}]")
            print(f"   目标：{m['description']}")
            print(f"   洛神当前：{m['luoshen_current']}")
            print(f"   需要环境：{m['environment_needed']}")
        
        s = results['summary']
        print(f"\n{'=' * 70}")
        print("差距统计")
        print(f"{'=' * 70}")
        print(f"  致命差距：{s['fatal']} 项")
        print(f"  严重差距：{s['severe']} 项")
        print(f"  中等差距：{s['moderate']} 项")
        print(f"  微小差距：{s['minor']} 项")
        
        print(f"\n{'=' * 70}")
        print("优先补充模块（致命→严重）")
        print(f"{'=' * 70}")
        for name, level, env in self.get_priority_modules():
            print(f"  [{level}] {name}")
            print(f"    → {env}")
        
        print(f"\n{'=' * 70}")


def main():
    analyzer = LanguageGapAnalyzer()
    analyzer.print_report()
    return analyzer


if __name__ == "__main__":
    main()