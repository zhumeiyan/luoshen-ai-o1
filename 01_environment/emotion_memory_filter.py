#!/usr/bin/env python3
"""
emotion_memory_filter.py — 情感-记忆滤镜
==============================

让情感状态直接影响回忆内容的筛选与排序。

核心机制：
1. 概念情感标签库：给常见概念打上情感极性标签
2. 情感兼容度计算：当前情感与每个回忆概念的兼容程度
3. 回忆重排序：根据兼容度重新排列激活概念
4. 情感驱动的线索选择：不同情感优先选择不同类型的回忆线索

规则：
- 开心 → 优先正面概念（太阳、花、朋友...）
- 难过 → 优先安慰概念（抱抱、妈妈、睡觉...）
- 愤怒 → 优先平静概念（风、云、水、安静...）
- 害怕 → 优先安全概念（家、妈妈、灯、温暖...）
- 平静 → 中性偏好，按原始激活度

这不是拟合，是真实的"情绪影响记忆提取"——就像人难过时会想起安慰的事。
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class EmotionMemoryFilter:
    """
    情感-记忆滤镜。
    
    根据当前情感状态，对回忆激活的概念进行筛选和重排序。
    """
    
    def __init__(self):
        # 概念情感标签库
        self.concept_tags = self._build_concept_tags()
        
        # 情感→优先标签映射
        self.emotion_preferences = {
            '开心':    ['正面', '快乐', '温暖', '社交'],
            '高兴':    ['正面', '快乐', '温暖', '社交'],
            '兴奋':    ['正面', '快乐', '活力'],
            '难过':    ['安慰', '温暖', '亲密', '正面'],
            '伤心':    ['安慰', '温暖', '亲密', '正面'],
            '委屈':    ['安慰', '亲密', '温暖'],
            '愤怒':    ['平静', '自然', '冷静', '正面'],
            '生气':    ['平静', '自然', '冷静', '正面'],
            '害怕':    ['安全', '安慰', '亲密', '温暖'],
            '恐惧':    ['安全', '安慰', '亲密'],
            '紧张':    ['平静', '安全', '自然'],
            '平静':    ['自然', '中性', '日常'],
            '无聊':    ['有趣', '活力', '正面'],
            '好奇':    ['有趣', '未知', '探索'],
            '惊讶':    ['有趣', '意外', '探索'],
            '困':      ['休息', '安静', '舒适'],
            '累':      ['休息', '安静', '舒适'],
            '饿':      ['食物', '安慰', '温暖'],
            '渴':      ['水', '清凉', '舒适'],
        }
        
        # 情感→避免标签映射
        self.emotion_avoids = {
            '开心':    ['负面', '危险', '悲伤'],
            '难过':    ['负面', '危险', '压力'],
            '愤怒':    ['负面', '刺激', '压力'],
            '害怕':    ['负面', '危险', '恐怖', '刺激'],
            '平静':    ['负面', '刺激', '压力'],
        }
    
    def _build_concept_tags(self) -> Dict[str, List[str]]:
        """构建概念情感标签库。"""
        return {
            # 正面/快乐
            '太阳': ['正面', '温暖', '自然', '光明'],
            '阳光': ['正面', '温暖', '自然', '光明'],
            '花': ['正面', '美丽', '自然', '快乐'],
            '花儿': ['正面', '美丽', '自然', '快乐'],
            '春天': ['正面', '温暖', '自然', '生机'],
            '星星': ['正面', '美丽', '夜晚', '梦幻'],
            '月亮': ['正面', '美丽', '夜晚', '安静'],
            '彩虹': ['正面', '美丽', '快乐', '惊喜'],
            '朋友': ['正面', '社交', '快乐', '温暖'],
            '玩具': ['正面', '快乐', '玩耍', '童年'],
            '苹果': ['正面', '食物', '甜美', '健康'],
            '牛奶': ['正面', '食物', '温暖', '健康'],
            '蛋糕': ['正面', '食物', '快乐', '甜美'],
            '糖': ['正面', '食物', '快乐', '甜美'],
            '笑': ['正面', '快乐', '表情', '社交'],
            '开心': ['正面', '快乐', '情感'],
            '高兴': ['正面', '快乐', '情感'],
            '喜欢': ['正面', '情感', '亲近'],
            '爱': ['正面', '情感', '亲密', '温暖'],
            '好玩': ['正面', '快乐', '玩耍'],
            '漂亮': ['正面', '美丽', '赞美'],
            '温暖': ['正面', '舒适', '情感', '安慰'],
            '明亮': ['正面', '光明', '舒适'],
            '香': ['正面', '感官', '舒适'],
            
            # 安慰/亲密
            '抱抱': ['安慰', '亲密', '温暖', '身体'],
            '妈妈': ['安慰', '亲密', '家庭', '温暖'],
            '爸爸': ['安慰', '亲密', '家庭', '温暖'],
            '家人': ['安慰', '亲密', '家庭', '温暖'],
            '睡觉': ['安慰', '休息', '舒适', '安静'],
            '休息': ['安慰', '休息', '舒适', '安静'],
            '家': ['安全', '安慰', '亲密', '温暖'],
            '床': ['安慰', '休息', '舒适', '安全'],
            '帮助': ['安慰', '支持', '社交', '正面'],
            '谢谢': ['安慰', '社交', '礼貌', '正面'],
            '好': ['正面', '安慰', '评价'],
            '乖': ['安慰', '亲密', '评价'],
            
            # 平静/自然
            '天空': ['平静', '自然', '广阔', '美丽'],
            '云': ['平静', '自然', '柔软', '美丽'],
            '风': ['平静', '自然', '清凉', '流动'],
            '树': ['平静', '自然', '生机', '稳定'],
            '草': ['平静', '自然', '柔软', '生机'],
            '水': ['平静', '自然', '清凉', '生命'],
            '海': ['平静', '自然', '广阔', '深邃'],
            '山': ['平静', '自然', '稳定', '高大'],
            '安静': ['平静', '舒适', '休息'],
            '慢慢': ['平静', '舒适', '节奏'],
            '轻轻': ['平静', '温柔', '舒适'],
            '软': ['平静', '舒适', '触觉'],
            
            # 安全/保护
            '灯': ['安全', '光明', '温暖', '夜晚'],
            '门': ['安全', '保护', '家'],
            '被子': ['安全', '温暖', '舒适', '休息'],
            '墙': ['安全', '保护', '稳定'],
            '警察': ['安全', '保护', '社会'],
            '医生': ['安全', '保护', '健康'],
            
            # 负面（难过时避免，但愤怒/害怕时可能用于表达）
            '害怕': ['负面', '恐惧', '情感'],
            '难过': ['负面', '悲伤', '情感'],
            '生气': ['负面', '愤怒', '情感'],
            '哭': ['负面', '悲伤', '表达'],
            '疼': ['负面', '痛苦', '身体'],
            '黑暗': ['负面', '恐惧', '未知'],
            '冷': ['负面', '不适', '身体'],
            '饿': ['负面', '不适', '身体'],
            '坏': ['负面', '评价', '危险'],
            '恐怖': ['负面', '恐惧', '危险'],
            '危险': ['负面', '恐惧', '警告'],
            '打': ['负面', '暴力', '疼痛'],
            '骂': ['负面', '暴力', '情感'],
            
            # 食物/安慰
            '饭': ['食物', '安慰', '温暖', '日常'],
            '菜': ['食物', '安慰', '温暖', '日常'],
            '肉': ['食物', '安慰', '能量'],
            '面': ['食物', '安慰', '温暖', '日常'],
            '粥': ['食物', '安慰', '温暖', '舒适'],
            '汤': ['食物', '安慰', '温暖', '舒适'],
            
            # 日常/中性
            '书': ['中性', '学习', '安静', '知识'],
            '画': ['中性', '艺术', '创造', '美丽'],
            '歌': ['中性', '艺术', '快乐', '表达'],
            '舞': ['中性', '艺术', '快乐', '身体'],
            '游戏': ['中性', '玩耍', '快乐', '竞争'],
            '球': ['中性', '玩耍', '运动', '竞争'],
            '车': ['中性', '工具', '移动', '现代'],
            '路': ['中性', '日常', '移动', '方向'],
        }
    
    def filter_concepts(self, activated: Dict[str, float], 
                        emotion: str,
                        top_k: int = 8) -> List[Tuple[str, float]]:
        """
        根据情感状态筛选和重排序激活概念。
        
        Args:
            activated: recall()返回的激活概念字典 {概念: 激活度}
            emotion: 当前主导情感
            top_k: 返回前k个概念
        
        Returns:
            筛选后的(概念, 兼容度分数)列表
        """
        if not activated:
            return []
        
        # 过滤虚词：这些词不应作为主题或情感对象
        stop_words = {'是', '的', '了', '在', '吗', '不', '很', '也', '都', '就',
                      '和', '有', '要', '想', '会', '能', '还', '让', '把', '被',
                      '给', '跟', '对', '从', '到', '向', '为', '与', '或',
                      '又', '再', '没', '别', '太', '真', '比较', '非常', '特别',
                      '有点', '那么', '这么', '什么', '怎么', '为什么',
                      '因为', '所以', '如果', '虽然', '但是', '然后', '最后',
                      '而且', '要么', '不但', '不仅',
                      '我', '你', '他', '她', '它', '我们', '你们', '他们',
                      '呀', '啊', '呢', '吧', '哦', '嗯', '哇', '哈',
                      '出', '来', '去', '上', '下', '前', '后', '里', '外',
                      '个', '只', '条', '张', '把', '根', '块', '片',
                      '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                      '这里', '那里', '这个', '那个', '这些', '那些'}
        
        filtered_activated = {k: v for k, v in activated.items() if k not in stop_words and len(k) >= 2}
        if not filtered_activated:
            filtered_activated = activated  # 如果过滤后为空，回退到原数据
        
        # 获取情感偏好和避免标签
        preferred_tags = self.emotion_preferences.get(emotion, ['中性'])
        avoid_tags = self.emotion_avoids.get(emotion, [])
        
        scored = []
        for concept, activation in filtered_activated.items():
            # 获取概念标签
            tags = self.concept_tags.get(concept, ['中性'])
            
            # 计算兼容度分数
            compatibility = self._compute_compatibility(
                tags, preferred_tags, avoid_tags
            )
            
            # 最终分数 = 激活度 * 兼容度
            final_score = activation * compatibility
            
            scored.append((concept, final_score, compatibility, activation))
        
        # 按最终分数排序
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 返回(概念, 兼容度)列表
        return [(c, compat) for c, _, compat, _ in scored[:top_k]]
    
    def _compute_compatibility(self, concept_tags: List[str],
                                preferred: List[str],
                                avoid: List[str]) -> float:
        """
        计算概念与当前情感的兼容度。
        
        返回 0.2 ~ 2.0 的兼容度系数：
        - 有偏好标签: +0.5 每个
        - 有避免标签: -0.5 每个（最低0.2）
        - 无标签: 1.0（中性）
        """
        score = 1.0
        
        # 偏好加分
        for tag in concept_tags:
            if tag in preferred:
                score += 0.5
        
        # 避免减分
        for tag in concept_tags:
            if tag in avoid:
                score -= 0.5
        
        # 限制范围
        return max(0.2, min(2.0, score))
    
    def get_emotion_cue(self, emotion: str, 
                        activated: Dict[str, float]) -> Optional[str]:
        """
        获取最适合当前情感状态的回忆线索。
        
        从情感过滤后的概念中选择最佳线索。
        """
        filtered = self.filter_concepts(activated, emotion, top_k=5)
        if filtered:
            return filtered[0][0]
        return None
    
    def get_emotion_helpers(self, emotion: str,
                            activated: Dict[str, float],
                            n_helpers: int = 4) -> List[str]:
        """
        获取适合当前情感的辅助概念列表。
        """
        filtered = self.filter_concepts(activated, emotion, top_k=n_helpers+2)
        
        # 返回概念列表（去掉线索词后的辅助概念）
        return [c for c, _ in filtered[1:n_helpers+1]]


def test_emotion_filter():
    """测试情感-记忆滤镜。"""
    print("=" * 60)
    print("情感-记忆滤镜测试")
    print("=" * 60)
    
    filter_engine = EmotionMemoryFilter()
    
    # 模拟回忆激活的概念
    mock_activated = {
        '太阳': 0.95,
        '害怕': 0.80,
        '妈妈': 0.75,
        '花': 0.70,
        '黑暗': 0.65,
        '抱抱': 0.60,
        '风': 0.55,
        '哭': 0.50,
        '温暖': 0.45,
        '朋友': 0.40,
    }
    
    test_emotions = ['开心', '难过', '愤怒', '害怕', '平静']
    
    print("\n模拟回忆激活概念:")
    for c, v in sorted(mock_activated.items(), key=lambda x: -x[1]):
        tags = filter_engine.concept_tags.get(c, ['中性'])
        print(f"  {c}: 激活度{v:.2f}, 标签{tags}")
    
    print("\n" + "-" * 60)
    
    for emotion in test_emotions:
        print(f"\n情感: {emotion}")
        print(f"  偏好: {filter_engine.emotion_preferences.get(emotion, ['中性'])}")
        print(f"  避免: {filter_engine.emotion_avoids.get(emotion, [])}")
        
        filtered = filter_engine.filter_concepts(mock_activated, emotion, top_k=5)
        
        print(f"  筛选后TOP5:")
        for concept, compat in filtered:
            orig_score = mock_activated[concept]
            final_score = orig_score * compat
            tags = filter_engine.concept_tags.get(concept, ['中性'])
            print(f"    {concept:6s} 原激活{orig_score:.2f} × 兼容{compat:.1f} = {final_score:.2f} {tags}")
        
        # 测试线索选择
        cue = filter_engine.get_emotion_cue(emotion, mock_activated)
        helpers = filter_engine.get_emotion_helpers(emotion, mock_activated, n_helpers=3)
        print(f"  推荐线索: {cue}")
        print(f"  推荐辅助: {helpers}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    test_emotion_filter()
