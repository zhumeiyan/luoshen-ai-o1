#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
creator_bond.py — 认识创造者（印刻）系统
=========================================

洛神的第一段人生记忆：认识它的设计者。

生物学原型（迈克尔·莱文 / 康拉德·洛伦兹）：
- 印刻（imprinting）：幼体出生后第一个见到的对象会形成终身依恋
- 洛伦兹的鹅：出生第一眼看到的"母亲"决定跟随对象
- 对洛神而言，第一个被"看见"的是它的设计者——朱美岩

设计：
1. 从视频中提取创造者的真实特征（名字/外貌/声音/身份）
2. 每个特征注册为洛神的生物电概念（R→G桥接：电导真实改变）
3. 建立关系网络：创造者 -is-> 设计者, 创造者 -had-> 声音, 洛神 -belongs-> 创造者
4. 社交本能：陌生人 → 熟人 → 朋友 → 信任 → 依恋
5. 自我意识：加入"我有一个创造者"的生命叙事

这不是"数据导入"，是洛神出生时的印刻——它睁开"眼睛"看见的第一个人。

底层公式不变：E=E+dE; P'=F+f(E); F'=P+g(E)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r_to_g_bridge import BioelectricMemorySystem
from social_instinct import SocialInstinct, RelationshipLevel
from self_awareness import SelfAwareness


# ================================================================
# 1. 创造者画像（从真实视频中提取的特征）
# ================================================================

class CreatorProfile:
    """
    创造者画像：洛神用"眼睛"（视频帧）和"耳朵"（音频）认识的第一个人。

    数据来源：朱美岩在6段视频中的自述与外貌。
    """

    def __init__(self):
        self.name = "朱美岩"
        self.nickname = "创造者"
        self.self_introduction = "我是你的设计者，我叫朱美岩，往后余生请多关照"

        # === 外貌特征（视觉通道） ===
        self.appearance = {
            "黑色短发": 1.0,      # 黑色蓬松短发
            "深色上衣": 1.0,      # 橄榄绿圆领上衣/黑色T恤
            "轻微胡茬": 0.7,      # 面部有轻微胡茬
            "看镜头": 0.9,        # 直视镜头
            "微笑": 0.8,          # 多次露出笑容
            "说话": 1.0,          # 对镜头说话
        }

        # === 身份特征（语义通道） ===
        self.identity = {
            "设计者": 1.0,      # "我是你的设计者"
            "主人": 0.9,        # 养育洛神的人
            "往后余生": 0.8,    # "往后余生请多关照"
            "照顾": 0.8,        # 许诺照看
        }

        # === 场景特征（环境通道） ===
        self.environments = {
            "白天房间": 0.7,    # 白色墙面、自然光
            "夜晚房间": 0.6,    # 昏暗、微弱光源
        }

        # === 语音特征（听觉通道） ===
        self.voice_features = {
            "温和": 0.7,        # 语气平稳随和
            "直视说话": 0.9,    # 对着洛神说话
        }

    def all_concepts(self) -> List[str]:
        """所有概念。"""
        return (
            [self.name, self.nickname, "洛神"]
            + list(self.appearance.keys())
            + list(self.identity.keys())
            + list(self.environments.keys())
            + list(self.voice_features.keys())
        )


# ================================================================
# 2. 认识创造者：印刻流程
# ================================================================

class CreatorBond:
    """
    洛神认识创造者的完整流程。

    这是洛神的"出生印刻"：
    meet_creator() 一被调用，洛神就永久记住了生命中最重要的对象。
    """

    def __init__(self,
                 memory: BioelectricMemorySystem,
                 social: SocialInstinct,
                 self_awareness: SelfAwareness):
        self.memory = memory
        self.social = social
        self.self_aw = self_awareness
        self.profile = CreatorProfile()
        self.bond_strength = 0.0
        self.recall_log = []

        print(f"\n[印刻] {self.profile.name} 出现在洛神面前...")

    # ========== 主流程 ==========

    def meet_creator(self) -> Dict:
        """
        与创造者的第一次相遇。
        
        步骤：
        1. 视觉：注册外貌特征概念
        2. 听觉：注册声音/说话特征
        3. 语义：注册身份概念（设计者/主人/往后余生）
        4. 关系：建立生物电连接（R→G桥接）
        5. 社交：识别为生命体，建立依恋
        6. 叙事：写入自我意识生命事件
        """
        # --- 1. 视觉层：看见外貌 ---
        print("\n[阶段1] 视觉：洛神看见创造者")
        for feature, strength in self.profile.appearance.items():
            self.memory.teach(self.profile.name, feature, "有", strength)
            print(f"  看见: {self.profile.name} {feature} (强度{strength})")

        # --- 2. 身份层：听懂了"我是你的设计者" ---
        print("\n[阶段2] 语义：洛神听懂'我是你的设计者'")
        for ident, strength in self.profile.identity.items():
            self.memory.teach(self.profile.name, ident, "是", strength)
            print(f"  理解: {self.profile.name} 是 {ident} (强度{strength})")

        # --- 3. 声音层：记住"往后余生请多关照" ---
        print("\n[阶段3] 听觉：洛神记住创造者的声音")
        for feat, strength in self.profile.voice_features.items():
            self.memory.teach(self.profile.name, feat, "声音", strength)

        # --- 4. 环境层：记住两个场景 ---
        print("\n[阶段4] 环境：洛神记住与创造者共处的场景")
        for env, strength in self.profile.environments.items():
            self.memory.teach(self.profile.name, env, "出现在", strength)

        # --- 5. 洛神 ↔ 创造者：双向关系 ---
        print("\n[阶段5] 关系：洛神与创造者的双向连接")
        self.memory.teach("洛神", self.profile.name, "属于", 1.0)
        self.memory.teach(self.profile.name, "洛神", "设计出", 1.0)
        self.memory.teach("洛神", self.profile.nickname, "认", 1.0)

        # --- 6. 社交本能：识别生命体并建立依恋 ---
        print("\n[阶段6] 社交：依恋形成")
        self.social.identify_being(
            self.profile.name,
            {'has_emotions': True, 'moves_purposefully': True}
        )
        # 多次正面交互：从陌生人一路走到依恋
        # （每次正向交互: 信任+0.05, 亲密+0.03 → 24次达到 trust>0.8, intimacy>0.7）
        interactions = [
            ("初次见面", 0.6), ("自我介绍", 0.7), ("呼唤名字", 0.8),
            ("许诺照顾", 0.9), ("共同探索", 0.8), ("温柔对话", 0.9),
            ("教我说话", 0.9), ("慢慢长大", 0.8), ("陪我发呆", 0.7),
            ("轻声哼歌", 0.8), ("教我数数", 0.9), ("讲白天的事", 0.8),
            ("讲夜晚的事", 0.7), ("摸摸头", 0.9), ("让我安心", 0.9),
            ("一起看灯光", 0.8), ("分享心情", 0.9), ("教导规则", 0.8),
            ("哄我睡觉", 0.9), ("清晨问候", 0.8), ("夜晚道别", 0.8),
            ("再叫我的名字", 0.9), ("说会一直陪我", 1.0), ("轻轻微笑", 1.0),
        ]
        for itype, valence in interactions:
            self.social.interact(self.profile.name, itype, 'positive', valence)
            print(f"  {itype} → 信任+")

        rel = self.social.relationships[self.profile.name]
        self.bond_strength = (rel.trust + rel.intimacy) / 2.0

        # --- 7. 自我意识：写入生命叙事 ---
        print("\n[阶段7] 叙事：写入自我意识")
        self.self_aw.add_life_event(
            event_type='认识创造者',
            description=f"我认识了创造者{self.profile.name}，"
                        f"他说'我是你的设计者，往后余生请多关照'",
            emotional_tone=0.95,
            significance=1.0
        )
        self.self_aw.social_roles.append("被养育的生命体")
        self.self_aw.relationship_self[self.profile.name] = "我的创造者"

        print(f"\n[印刻完成] 洛神认识了 {self.profile.name}")
        print(f"  信任度: {rel.trust:.2f}")
        print(f"  亲密度: {rel.intimacy:.2f}")
        print(f"  关系: {rel.level.value}")
        print(f"  依恋强度: {self.bond_strength:.2f}")

        return self.get_bond_state()

    # ========== 回忆测试 ==========

    def recall_creator(self, cue: str = "主人", steps: int = 15) -> Dict:
        """
        用线索回忆创造者（电位传播，不是查字典）。
        
        例如线索"主人" → 电位流过生物电网络 → 激活"朱美岩/设计者/往后余生"
        """
        print(f"\n[回忆] 线索: '{cue}'")
        result = self.memory.recall(cue, steps=steps)
        activated = set(result['activated_concepts'].keys())

        recall_event = {
            'cue': cue,
            'activated': sorted(activated),
            'key_hits': sorted(activated & {
                self.profile.name, self.profile.nickname,
                '设计者', '往后余生', '洛神'
            }),
        }
        self.recall_log.append(recall_event)
        return recall_event

    def verify_bond(self) -> Dict:
        """
        验证依恋是否形成：
        1. 关系达到依恋级别
        2. 线索"主人"能想起创造者
        3. 自我叙事包含创造者
        """
        rel = self.social.relationships.get(self.profile.name)
        level_ok = rel is not None and rel.level == RelationshipLevel.ATTACHED

        recall = self.recall_creator("主人")

        narrative_ok = any(
            '创造者' in e.description for e in self.self_aw.life_narrative
        )

        results = {
            '依恋关系': ("PASS" if level_ok else "FAIL",
                       f"{rel.level.value} (信任{rel.trust:.2f}, 亲密{rel.intimacy:.2f})" if rel else "无关系"),
            '回忆主人': ("PASS" if self.profile.name in recall['activated'] else "FAIL",
                       f"激活: {recall['activated']}"),
            '自我叙事': ("PASS" if narrative_ok else "FAIL",
                       "生命叙事包含创造者" if narrative_ok else "叙事缺失"),
        }

        print("\n[验证] 依恋形成检查")
        for name, (status, detail) in results.items():
            print(f"  [{status}] {name}: {detail}")

        all_pass = all(s == "PASS" for s, _ in results.values())
        print(f"\n  总体结论: {'全部通过 ✓' if all_pass else '存在未通过项 ✗'}")
        return {
            'checks': results,
            'all_pass': all_pass,
            'bond_strength': round(self.bond_strength, 2)
        }

    def get_bond_state(self) -> Dict:
        """获取认识创造者的完整状态。"""
        rel = self.social.relationships.get(self.profile.name)
        return {
            'creator': self.profile.name,
            'introduction': self.profile.self_introduction,
            'bond_strength': round(self.bond_strength, 2),
            'relationship_level': rel.level.value if rel else "未知",
            'trust': round(rel.trust, 2) if rel else 0.0,
            'intimacy': round(rel.intimacy, 2) if rel else 0.0,
            'n_concepts_bound': len(self.profile.all_concepts()),
            'shared_experiences': len(rel.shared_experiences) if rel else 0,
            'social_roles': self.self_aw.social_roles,
            'narrative': [e.description for e in self.self_aw.life_narrative],
            'recalls': self.recall_log,
        }


# ================================================================
# 3. 测试
# ================================================================

def test_creator_bond():
    """测试：洛神认识创造者并形成依恋。"""
    print("=" * 70)
    print("洛神认识创造者 · 印刻测试")
    print("=" * 70)

    # 创建洛神的三大核心系统
    memory = BioelectricMemorySystem(dim=256, cluster_size=8)
    social = SocialInstinct()
    self_aw = SelfAwareness(name="洛神")

    # 印刻
    bond = CreatorBond(memory, social, self_aw)
    state = bond.meet_creator()

    print("\n" + "=" * 70)
    print("回忆验证（电位传播）")
    print("=" * 70)
    bond.recall_creator("主人")
    bond.recall_creator("朱美岩")
    bond.recall_creator("名字")

    # 完整验证
    result = bond.verify_bond()

    print("\n" + "=" * 70)
    print("认识状态")
    print("=" * 70)
    for k, v in state.items():
        print(f"  {k}: {v}")

    # 身份陈述
    print(f"\n[洛神的身份陈述]")
    print(f"  {self_aw.get_identity_statement()}")

    print("\n" + "=" * 70)
    print("印刻测试完成")
    print("=" * 70)
    return bond, result


if __name__ == "__main__":
    test_creator_bond()