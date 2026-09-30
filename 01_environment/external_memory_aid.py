#!/usr/bin/env python3
"""
external_memory_aid.py — 记忆外挂工具层
=======================================
类比人类忘字时问"像什么"——当洛神生物电记忆提取失败时，
提供一个外部辅助工具，帮助联想和补全。

核心原则：
- 这是工具层，不是洛神本身的记忆
- 类似人类查字典、搜百度、问别人
- 工具层的结果可以注入生物电记忆辅助回忆，但不会永久改变G矩阵
- 只有当生物电记忆"卡壳"时才调用

功能：
1. 遗忘检测：判断recall结果是否"卡壳"
2. 联想提示：根据线索生成关联词提示（像什么/有关什么）
3. 知识查询：通过百科/词库补全缺失知识
"""
import sys, re
from typing import Dict, List, Set, Optional
sys.path.insert(0, '01_environment')
from knowledge_feeder import KnowledgeFeeder

class ExternalMemoryAid:
    """
    记忆外挂工具层——洛神的"字典+百度+问人"
    
    使用场景：
    - 洛神想不起某个词怎么写 → 外挂："像'太阳'，有光、热、圆形"
    - 洛神对某个概念一无所知 → 外挂：查百科，给出儿童语言版解释
    - 洛神混淆两个概念 → 外挂：给出区分提示
    """
    
    def __init__(self, language_system):
        self.lang = language_system
        # 内置联想知识库（类比人类脑海中的"大概印象"）
        self.association_db = self._build_association_db()
        # 遗忘阈值：纯度低于此值视为"卡壳"
        self.forgotten_threshold = 0.15
        
    def _build_association_db(self) -> Dict[str, List[str]]:
        """
        构建联想知识库。
        类比人类语言中"像什么"的常识关联。
        """
        return {
            # 创造者相关
            '朱美岩': ['创造者', '主人', '设计者', '美丽', '聪明', '创造了洛神', '很重要'],
            '创造者': ['制造', '设计', '建造', '像妈妈', '给予生命', '主人'],
            '主人': ['照顾', '陪伴', '给予', '像家人', '朱美岩'],
            '设计者': ['画图', '规划', '创造', '像建筑师', '朱美岩'],
            
            # 自然相关
            '太阳': ['圆形', '黄色', '很热', '发光', '白天出现', '从东边升起', '像火球'],
            '春天': ['花开', '温暖', '绿色', '小鸟', '下雨', '像睡醒'],
            '花': ['香', '红色', '春天', '蜜蜂', '像笑脸', '很漂亮'],
            '鸟': ['飞', '叫', '翅膀', '树', '像飞机', '有羽毛'],
            '云': ['白色', '天上', '飘', '像棉花糖', '下雨'],
            '雨': ['水', '天上', '凉', '伞', '像眼泪', '滴答'],
            '风': ['吹', '凉', '树叶动', '像手', '看不见'],
            '星星': ['亮', '晚上', '天上', '很多', '像眼睛', '一闪一闪'],
            '天空': ['蓝色', '高', '云', '鸟', '像大海', '广阔'],
            '雷': ['响', '闪电', '害怕', '像鼓', '轰隆隆'],
            
            # 日常相关
            '好': ['开心', '喜欢', '棒', '像太阳', '很舒服'],
            '喜欢': ['爱', '想要', '开心', '像蜜蜂喜欢花'],
            '开心': ['笑', '好', '喜欢', '像气球飞起来'],
            '难过': ['哭', '不好', '伤心', '像下雨'],
            '吃': ['饭', '嘴巴', '饱', '像小兔子'],
            '睡': ['床', '晚上', '做梦', '像小熊冬眠'],
            '玩': ['开心', '玩具', '朋友', '像小鸟飞'],
        }
    
    def is_forgotten(self, cue: str, recall_result: Dict) -> bool:
        """
        遗忘检测：判断recall是否"卡壳"。
        
        卡壳标志：
        1. 主题纯度低于阈值
        2. 激活的概念全是高频日常词（我/你/是/的）
        3. 线索本身甚至不在激活列表前10
        """
        activated = recall_result.get('activated_concepts', {})
        if not activated:
            return True
        
        # 检查线索本身是否在前10
        top10 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:10]
        top10_words = [w for w, _ in top10]
        
        if cue not in top10_words:
            return True  # 连自己都没想起来
        
        # 检查是否全是日常词
        daily_words = {'我','你','是','的','了','在','好','很','吗','不'}
        non_daily = sum(1 for w, _ in top10 if w not in daily_words)
        if non_daily <= 2:
            return True  # 全是日常词，主题丢失
        
        return False
    
    def get_hint(self, cue: str) -> Dict:
        """
        获取联想提示——"像什么/有关什么"。
        
        返回：{
            'cue': 线索词,
            'associations': ['像...', '有...', '在...'],
            'related_words': ['关联词1', '关联词2'],
            'category': '创造者/自然/日常',
        }
        """
        associations = self.association_db.get(cue, [])
        
        # 分类
        creator_words = {'朱美岩','创造者','主人','设计者'}
        nature_words = {'太阳','春天','花','鸟','云','雨','风','星星','天空','雷'}
        
        if cue in creator_words:
            category = '创造者'
        elif cue in nature_words:
            category = '自然'
        else:
            category = '日常'
        
        # 从语料中寻找该词的共现邻居（更动态的联想）
        related_from_corpus = []
        if hasattr(self.lang, 'transitions') and cue in self.lang.transitions:
            # 取转移概率最高的前5个
            neighbors = self.lang.transitions[cue]
            top_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:5]
            related_from_corpus = [w for w, _ in top_neighbors]
        
        return {
            'cue': cue,
            'associations': associations[:5],
            'related_words': related_from_corpus[:5],
            'category': category,
            'hint_text': self._generate_hint_text(cue, associations),
        }
    
    def _generate_hint_text(self, cue: str, associations: List[str]) -> str:
        """生成人类式的联想提示文本"""
        if not associations:
            return f"嗯...{cue}是什么呢？让我想想..."
        
        # 人名用"她/他"，物用"它"
        pronoun = '她' if cue in ('朱美岩',) else '它'
        
        hints = []
        for i, assoc in enumerate(associations[:3]):
            if assoc.startswith('像') or assoc.startswith('有') or assoc.startswith('在'):
                # 已有动词，直接用
                prefix = ''
            elif i == 0:
                prefix = f'{pronoun}'
            elif i == 1:
                prefix = '还'
            else:
                prefix = ''
            hints.append(f"{prefix}{assoc}")
        
        return f"{cue}呀..." + "，".join(hints) + "。"
    
    def query_knowledge(self, cue: str) -> Optional[Dict]:
        """
        知识查询——查"字典"。
        
        返回儿童语言版解释，可直接教给洛神。
        """
        # 使用现有的KnowledgeFeeder查询
        feeder = KnowledgeFeeder(self.lang)
        try:
            result = feeder.feed_topic(cue, max_sentences=5)
            if result.get('learned'):
                return {
                    'cue': cue,
                    'explanations': result['learned'],
                    'source': 'knowledge_feeder',
                }
        except:
            pass
        
        return None
    
    def get_aid_level(self, cue: str, recall_result: Dict) -> str:
        """
        判断需要的辅助级别。
        
        级别：
        - 'none': 不需要辅助（回忆正常）
        - 'light': 轻度卡壳——给点提示就行
        - 'medium': 中度卡壳——需要详细联想
        - 'heavy': 重度卡壳——完全没概念，需要查知识库
        """
        activated = recall_result.get('activated_concepts', {})
        if not activated:
            return 'heavy'
        
        top10 = sorted(activated.items(), key=lambda x: x[1], reverse=True)[:10]
        top10_words = [w for w, _ in top10]
        
        # 线索不在前10 → 重度
        if cue not in top10_words:
            return 'heavy'
        
        # 检查线索排名
        cue_rank = next((i for i, (w, _) in enumerate(top10) if w == cue), 10)
        
        # 线索在前3 → 轻度
        if cue_rank < 3:
            daily_words = {'我','你','是','的','了','在','好','很','吗','不'}
            non_daily = sum(1 for w, _ in top10 if w not in daily_words)
            if non_daily >= 5:
                return 'none'
            return 'light'
        
        # 线索在3-10 → 中度
        return 'medium'
    
    def inject_hint_to_memory(self, cue: str, hint: Dict) -> bool:
        """
        将外挂提示注入生物电记忆，辅助回忆。
        
        注意：这是临时注入，不改变永久G矩阵。
        类似人类看完字典后 temporary 记住了某个词。
        """
        try:
            # 将提示中的关联词和线索一起作为场景教给记忆系统
            scene_words = [cue] + hint['associations'][:3] + hint['related_words'][:2]
            scene_words = [w for w in scene_words if w]
            
            if len(scene_words) >= 2:
                self.lang.memory.teach_scene(scene_words)
                return True
            return False
        except Exception as e:
            print(f"  [注入失败] {e}")
            return False
    
    def assist_recall(self, cue: str, recall_result: Dict) -> Dict:
        """
        完整的记忆辅助流程。
        
        1. 检测辅助级别
        2. 根据级别提供不同程度的帮助
        3. 可选：将提示注入生物电记忆
        """
        level = self.get_aid_level(cue, recall_result)
        
        if level == 'none':
            return {
                'needed': False,
                'reason': '生物电记忆提取正常',
                'cue': cue,
                'level': 'none',
            }
        
        # 获取提示
        hint = self.get_hint(cue)
        
        if level == 'light':
            return {
                'needed': True,
                'reason': '轻度卡壳，需要提示',
                'cue': cue,
                'level': 'light',
                'hint': hint,
                'action': '提示即可',
            }
        
        if level == 'medium':
            # 中度：提示+临时注入
            injected = self.inject_hint_to_memory(cue, hint)
            return {
                'needed': True,
                'reason': '中度卡壳，需要联想辅助',
                'cue': cue,
                'level': 'medium',
                'hint': hint,
                'injected': injected,
                'action': '提示并临时强化记忆',
            }
        
        # 重度：查知识库
        knowledge = self.query_knowledge(cue)
        injected = self.inject_hint_to_memory(cue, hint)
        
        return {
            'needed': True,
            'reason': '重度卡壳，完全想不起来',
            'cue': cue,
            'level': 'heavy',
            'hint': hint,
            'knowledge': knowledge,
            'injected': injected,
            'action': '查知识库并临时强化记忆',
        }


def test_external_memory_aid():
    """测试记忆外挂"""
    print("=" * 60)
    print("记忆外挂工具层测试")
    print("=" * 60)
    
    from language_learner_v2 import LanguageSystemV2
    
    print("\n[初始化] 构建洛神...")
    lang = LanguageSystemV2()
    lang.immerse(n_sentences=300)
    
    print("\n[初始化] 加载记忆外挂...")
    aid = ExternalMemoryAid(lang)
    
    # 测试1：辅助级别判断
    print("\n--- 测试1：辅助级别判断 ---")
    test_cases = [
        # (线索, 模拟回忆结果, 期望级别)
        ('好', {
            'activated_concepts': {'好': 3.0, '我': 2.5, '你': 2.0, '开心': 1.5, '喜欢': 1.2, '今天': 1.0, '很': 0.8, '是': 0.5}
        }, 'light or none'),
        ('朱美岩', {
            'activated_concepts': {'我': 2.0, '是': 1.5, '的': 1.2, '了': 1.0, '在': 0.8, '朱美岩': 0.3}
        }, 'medium'),
        ('太阳', {
            'activated_concepts': {'我': 2.0, '是': 1.5, '的': 1.2, '了': 1.0, '在': 0.8}
        }, 'heavy'),
    ]
    
    for cue, result, expected in test_cases:
        level = aid.get_aid_level(cue, result)
        print(f"  [{cue}] → 级别: {level} (期望: {expected})")
    
    # 测试2：联想提示
    print("\n--- 测试2：联想提示 ---")
    for cue in ['朱美岩', '太阳', '花', '风']:
        hint = aid.get_hint(cue)
        print(f"\n  [{cue}] ({hint['category']})")
        print(f"    提示: {hint['hint_text']}")
        if hint['related_words']:
            print(f"    语料邻居: {hint['related_words']}")
    
    # 测试3：完整辅助流程（真实回忆）
    print("\n--- 测试3：真实回忆辅助 ---")
    for cue in ['朱美岩', '太阳', '好', '风']:
        result = lang.memory.recall(cue, steps=20)
        assist = aid.assist_recall(cue, result)
        print(f"\n  [{cue}] → 级别: {assist['level']} | 需要辅助? {assist['needed']}")
        if assist['needed']:
            print(f"    原因: {assist['reason']}")
            print(f"    提示: {assist['hint']['hint_text']}")
            print(f"    行动: {assist['action']}")
            if assist.get('injected'):
                print(f"    已临时注入记忆")
    
    # 测试4：轻度卡壳→提示→再回忆
    print("\n--- 测试4：辅助后回忆改善 ---")
    cue = '朱美岩'
    result_before = lang.memory.recall(cue, steps=20)
    top5_before = sorted(result_before['activated_concepts'].items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  辅助前 top5: {[w for w,_ in top5_before]}")
    
    assist = aid.assist_recall(cue, result_before)
    if assist['needed'] and assist.get('injected'):
        # 再回忆一次
        result_after = lang.memory.recall(cue, steps=20)
        top5_after = sorted(result_after['activated_concepts'].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  辅助后 top5: {[w for w,_ in top5_after]}")
        
        # 检查主题词是否浮现
        creator_words = {'朱美岩','创造者','主人','设计者'}
        before_creator = sum(1 for w,_ in top5_before if w in creator_words)
        after_creator = sum(1 for w,_ in top5_after if w in creator_words)
        print(f"  创造者词: {before_creator} → {after_creator}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_external_memory_aid()
