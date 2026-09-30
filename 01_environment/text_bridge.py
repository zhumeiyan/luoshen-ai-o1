"""
text_bridge.py — 文本输入桥接
===========================
人类语言 → 概念电位 → 感官注入

流程：
1. 分词（jieba or 简单空格/标点分割）
2. 查概念库（已知概念→直接编码）
3. 未知词→动态注册→自动编码
4. 多词组合→生成dE叠加
"""
import numpy as np
import re
from expandable_concepts import ExpandableConceptSpace

class TextBridge:
    """文本→生物电桥接"""
    
    def __init__(self, concept_space=None):
        self.cs = concept_space or ExpandableConceptSpace()
    
    def tokenize(self, text):
        """简单中文分词：连续中文字符按2字切分，英文按空格"""
        """简单分词：按标点/空格分割"""
        # 移除标点，按空格和常见分隔符分
        # 先切分中文字符（每2字一组）
        segments = []
        i = 0
        while i < len(text):
            if text[i].isalpha():
                # 收集连续字母或汉字
                j = i
                while j < len(text) and text[j].isalpha():
                    j += 1
                word = text[i:j]
                if len(word) >= 2 and all(ord(c) > 127 for c in word):
                    # 纯中文，按2字切
                    for k in range(0, len(word), 2):
                        segments.append(word[k:k+2])
                else:
                    segments.append(word)
                i = j
            else:
                i += 1
        return [s for s in segments if len(s.strip()) > 0]

        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [t.strip() for t in text.split() if t.strip()]
        return tokens
    
    def text_to_potential(self, text):
        """
        文本 → 概念电位向量
        
        每个词贡献自己的8维哈希编码，叠加形成总dE。
        未知词自动注册。
        """
        tokens = self.tokenize(text)
        dE = np.zeros(self.cs.dim)
        
        for token in tokens:
            # 自动注册（未知词）或获取（已知词）
            self.cs.get_id(token)
            # 叠加编码
            dE += self.cs.encode(token)
        
        # 归一化（避免过长文本导致电位爆炸）
        if np.max(np.abs(dE)) > 5:
            dE = dE / np.max(np.abs(dE)) * 5
        
        return dE, tokens
    
    def interpret_response(self, active_concepts, top_n=5):
        """
        反向：激活的概念ID → 人类可理解的词
        
        返回最活跃的top_n个概念名
        """
        # 按激活度排序
        sorted_ids = sorted(active_concepts, key=lambda x: abs(x) if isinstance(x, (int, float, np.number)) else 0, reverse=True)
        
        # ID→名称映射
        id2name = {v["id"]: k for k, v in self.cs.concepts.items()}
        
        names = []
        for cid in sorted_ids[:top_n]:
            if isinstance(cid, (int, np.integer)) and cid in id2name:
                names.append(id2name[cid])
        
        return names


if __name__ == "__main__":
    print("=" * 60)
    print("文本桥接测试")
    print("=" * 60)
    
    bridge = TextBridge()
    
    print("\n=== T1 简单句子 → 电位 ===")
    text = "乌鸦口渴了放下石子"
    dE, tokens = bridge.text_to_potential(text)
    print(f"  输入: '{text}'")
    print(f"  分词: {tokens}")
    print(f"  电位向量非零维度: {np.count_nonzero(dE)}")
    print(f"  电位最大值: {np.max(dE):.2f}")
    
    print("\n=== T2 未知词自动注册 ===")
    text2 = "人工智能正在学习量子计算"
    dE2, tokens2 = bridge.text_to_potential(text2)
    print(f"  输入: '{text2}'")
    print(f"  分词: {tokens2}")
    print(f"  概念库总数: {bridge.cs.stats()['total_concepts']}")
    
    print("\n=== T3 长文本归一化 ===")
    text3 = "乌鸦 乌鸦 乌鸦 乌鸦 乌鸦"  # 重复词
    dE3, _ = bridge.text_to_potential(text3)
    print(f"  5个'乌鸦'叠加后最大值: {np.max(dE3):.2f} (应≤5)")
    assert np.max(dE3) <= 5.1, "长文本应归一化"
    print("  [PASS] 归一化成功")
    
    print("\n" + "=" * 60)
    print("文本桥接测试全部通过 ✓")
    print("=" * 60)
