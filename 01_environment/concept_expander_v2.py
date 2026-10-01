#!/usr/bin/env python3
"""
concept_expander_v2.py — 智能概念扩展器
=======================================

问题：G矩阵概念空间已满（512上限）。
解决：像人脑一样——不是无限容量，而是复用和压缩。

策略：
1. 概念复用：新词和已有概念相似时，映射到已有神经元（不注册新概念）
2. 低频遗忘：空间满时，替换最低频的旧概念
3. 复合概念：多字词映射到已有单字概念的叠加（如"大海"→"大"+"海"）

生物学类比：
- 概念复用 = 一词多义（同一个神经元集群参与多个概念）
- 低频遗忘 = 记忆消退（不常用的记忆淡化）
- 复合概念 = 语义组合（"红"+"花"→"红花"）
"""

import sys, json
from typing import Dict, List, Set, Optional
from collections import defaultdict

sys.path.insert(0, '01_environment')
from language_learner_v2 import LanguageSystemV2


class SmartConceptExpander:
    """
    智能概念扩展器。
    
    在有限空间内最大化概念覆盖。
    """
    
    def __init__(self, lang_system: LanguageSystemV2):
        self.lang = lang_system
        self.max_concepts = 512
        
        # 概念复用映射：新词 → 已有概念（相似词）
        self.concept_aliases = {}
        
        # 复合概念规则：多字词 → 单字组合
        self.composite_rules = {}
        
        # 遗忘阈值：低于此频次的概念可被替换
        self.forget_threshold = 2
        
        self._build_alias_map()
    
    def _build_alias_map(self):
        """构建概念别名映射"""
        # 同义词/近义词映射
        self.concept_aliases = {
            '大海': '海',
            '森林': '林',
            '宇宙': '空',
            '地球': '地',
            '月亮': '月',
            '星星': '星',
            '太阳': '阳',
            '彩虹': '彩',
            '雪花': '雪',
            '冰川': '冰',
            '沙漠': '沙',
            '草原': '草',
            '瀑布': '水',
            '温泉': '泉',
            '洞穴': '洞',
            '岛屿': '岛',
            
            '光速': '光',
            '声音': '声',
            '植物': '花',
            '动物': '鸟',
            '能量': '力',
            '磁场': '磁',
            '电流': '电',
            '摩擦': '摩',
            
            '中国': '国',
            '长城': '城',
            '黄河': '河',
            '故宫': '宫',
            '丝绸': '丝',
            '造纸': '纸',
            '印刷': '印',
            '火药': '药',
            '指南': '南',
            '端午': '午',
            '中秋': '秋',
            '春节': '春',
            '书法': '书',
            '京剧': '剧',
            '水墨': '墨',
            '古筝': '琴',
            '围棋': '棋',
            '象棋': '棋',
            '太极': '极',
            '中医': '医',
            '道德': '德',
            '孟子': '孟',
            '庄子': '庄',
            '史记': '史',
            
            '音乐': '乐',
            '画画': '画',
            '舞蹈': '舞',
            '雕塑': '雕',
            '电影': '影',
            '摄影': '影',
            '建筑': '筑',
            '诗歌': '诗',
            '小说': '说',
            '戏剧': '戏',
            '歌剧': '歌',
            '芭蕾': '舞',
            '油画': '画',
            '素描': '描',
            '水彩': '彩',
            '版画': '画',
            '钢琴': '琴',
            '小提琴': '琴',
            '吉他': '琴',
            '笛子': '笛',
            '二胡': '琴',
            '唢呐': '呐',
            '琵琶': '琴',
            
            '早餐': '餐',
            '午餐': '餐',
            '晚餐': '餐',
            '水果': '果',
            '运动': '动',
            '睡觉': '睡',
            '刷牙': '牙',
            '洗手': '手',
            '穿衣': '衣',
            '走路': '走',
            '骑车': '车',
            '坐车': '车',
            '读书': '读',
            '写字': '写',
            '唱歌': '唱',
            '跳舞': '跳',
            '做饭': '饭',
            '洗碗': '洗',
            '扫地': '扫',
            '擦窗': '窗',
            '浇花': '浇',
            '遛狗': '狗',
            '购物': '购',
            '旅行': '旅',
            
            '思念': '想',
            '期待': '待',
            '担心': '怕',
            '骄傲': '傲',
            '尴尬': '尬',
            '失望': '望',
            '希望': '望',
            '感激': '感',
            '原谅': '原',
            '嫉妒': '妒',
            '同情': '情',
            '孤独': '独',
            '平静': '静',
            '兴奋': '兴',
            '紧张': '张',
            '安心': '安',
            '满足': '足',
            '好奇': '奇',
            '困惑': '困',
            
            '问候': '问',
            '告别': '别',
            '感谢': '谢',
            '道歉': '歉',
            '赞美': '美',
            '批评': '评',
            '商量': '商',
            '合作': '合',
            '分享': '分',
            '借东': '借',
            '守时': '时',
            '诚实': '诚',
            '倾听': '听',
            '表达': '表',
            '微笑': '笑',
            '握手': '握',
            '拥抱': '抱',
            '礼物': '礼',
            '聚会': '聚',
            '争吵': '争',
            '友谊': '友',
            '家庭': '家',
            '团队': '团',
            '社区': '区',
        }
    
    def smart_tokenize(self, text: str) -> List[str]:
        """
        智能分词：优先复用已有概念。
        
        策略：
        1. 如果整词在概念库中，直接返回
        2. 如果整词有别名映射，返回别名
        3. 如果整词是复合词，尝试拆分为已有单字
        4. 如果空间未满，注册新概念
        5. 如果空间已满，替换最低频概念
        """
        tokens = self.lang.tokenizer.tokenize(text)
        smart_tokens = []
        
        for token in tokens:
            # 1. 已在概念库中
            if token in self.lang.learned_words:
                smart_tokens.append(token)
                continue
            
            # 2. 有别名映射
            if token in self.concept_aliases:
                alias = self.concept_aliases[token]
                if alias in self.lang.learned_words:
                    smart_tokens.append(alias)
                    continue
            
            # 3. 尝试注册新概念
            registered = self._try_register_concept(token)
            if registered:
                smart_tokens.append(token)
            else:
                # 4. 空间满，尝试替换
                replaced = self._try_replace_concept(token)
                if replaced:
                    smart_tokens.append(token)
                else:
                    # 5. 无法注册，跳过或映射到单字
                    if len(token) >= 2:
                        # 拆分为单字
                        for char in token:
                            if char in self.lang.learned_words:
                                smart_tokens.append(char)
                    # 单字也无法映射，跳过
        
        return smart_tokens
    
    def _try_register_concept(self, concept: str) -> bool:
        """尝试注册新概念，如果空间未满"""
        if len(self.lang.learned_words) < self.max_concepts:
            return True
        return False
    
    def _try_replace_concept(self, new_concept: str) -> bool:
        """
        尝试替换最低频旧概念。
        
        返回是否成功替换。
        """
        if not self.lang.learned_words:
            return False
        
        # 找到最低频的概念
        min_freq = float('inf')
        min_concept = None
        
        for concept, freq in self.lang.learned_words.items():
            if freq < min_freq and freq <= self.forget_threshold:
                min_freq = freq
                min_concept = concept
        
        if min_concept:
            # 遗忘旧概念
            del self.lang.learned_words[min_concept]
            print(f"  [遗忘] '{min_concept}' (频次{min_freq}) → 为新概念让位")
            return True
        
        return False
    
    def expand_by_smart_corpus(self, sentences: List[str]) -> Dict:
        """
        通过智能语料浸泡扩展概念。
        
        在有限空间内最大化覆盖。
        """
        print(f"[智能扩展] 开始语料浸泡: {len(sentences)}句")
        
        stats_before = len(self.lang.learned_words)
        registered_new = 0
        reused = 0
        replaced = 0
        
        for sentence in sentences:
            tokens = self.smart_tokenize(sentence)
            
            # 学习统计
            for token in tokens:
                was_new = token not in self.lang.learned_words
                self.lang.learned_words[token] += 1
                
                if was_new:
                    if len(self.lang.learned_words) > stats_before + registered_new:
                        # 这是替换产生的"新"概念
                        replaced += 1
                    else:
                        registered_new += 1
                else:
                    reused += 1
            
            # 共现学习
            for j in range(len(tokens)):
                for k in range(j+1, min(j+4, len(tokens))):
                    t1, t2 = tokens[j], tokens[k]
                    self.lang.learned_pairs += 1
        
        stats_after = len(self.lang.learned_words)
        
        return {
            'sentences_processed': len(sentences),
            'concepts_before': stats_before,
            'concepts_after': stats_after,
            'registered_new': registered_new,
            'reused': reused,
            'replaced': replaced,
        }
    
    def get_expansion_report(self) -> Dict:
        """生成扩展报告"""
        return {
            'total_concepts': len(self.lang.learned_words),
            'total_pairs': self.lang.learned_pairs,
            'space_usage': f"{len(self.lang.learned_words)}/{self.max_concepts}",
            'top_words': sorted(self.lang.learned_words.items(), key=lambda x: x[1], reverse=True)[:20],
        }


def test_smart_expansion():
    """测试智能概念扩展"""
    print("=" * 60)
    print("智能概念扩展测试")
    print("=" * 60)
    
    # 初始化洛神
    print("\n[初始化] 构建洛神...")
    lang = LanguageSystemV2()
    lang.immerse(n_sentences=300)
    
    initial_concepts = len(lang.learned_words)
    print(f"初始状态: {initial_concepts}概念/{lang.learned_pairs}连接")
    
    # 构建扩展语料
    expander = SmartConceptExpander(lang)
    corpus = expander.lang.env.expose(1000)  # 从环境获取大量句子
    
    # 智能扩展
    print(f"\n[扩展] 开始智能语料浸泡...")
    result = expander.expand_by_smart_corpus(corpus)
    
    print(f"\n扩展结果:")
    print(f"  处理句子: {result['sentences_processed']}")
    print(f"  概念变化: {result['concepts_before']} → {result['concepts_after']}")
    print(f"  新增注册: {result['registered_new']}")
    print(f"  复用已有: {result['reused']}")
    print(f"  替换遗忘: {result['replaced']}")
    
    # 报告
    print(f"\n[报告] 扩展后状态:")
    report = expander.get_expansion_report()
    print(f"  总概念: {report['total_concepts']}")
    print(f"  空间使用: {report['space_usage']}")
    print(f"  高频词TOP10:")
    for word, count in report['top_words'][:10]:
        print(f"    {word}: {count}次")
    
    # 保存
    with open('smart_expansion_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n[报告] smart_expansion_report.json")
    print("=" * 60)


if __name__ == '__main__':
    test_smart_expansion()
