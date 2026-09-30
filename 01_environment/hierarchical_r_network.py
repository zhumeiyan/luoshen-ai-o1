"""
hierarchical_r_network.py — 层次化R网（故事→场景→概念）
===============================================
不靠大向量，靠多层R网解决长程上下文。

三层结构：
  故事层（Story）: 乌鸦喝水 → {场景A, 场景B, 场景C}
  场景层（Scene）: 场景A → {乌鸦, 石子, 水}
  概念层（Concept）: 乌鸦 → {是:鸟, 能够:飞, 需要:水}

回忆传播：线索 → 概念层 → 场景层 → 故事层 → 跨场景联想
"""
import numpy as np
from collections import defaultdict

class HierarchicalRNetwork:
    """三层R网：故事 → 场景 → 概念"""
    
    def __init__(self):
        # 概念层（原子概念之间的关系）
        self.concept_R = defaultdict(list)  # (subj, obj) -> [(rel, w)]
        
        # 场景层（场景→包含哪些概念）
        self.scene_concepts = {}  # scene_id -> {concept_ids}
        self.scene_labels = {}    # scene_id -> label
        
        # 故事层（故事→包含哪些场景）
        self.story_scenes = {}    # story_id -> {scene_ids}
        self.story_labels = {}    # story_id -> label
    
    # ========== 概念层（最底层）==========
    def add_concept_rel(self, subj, obj, rel, w=1.0):
        """添加概念层关系"""
        key = (subj, obj)
        self.concept_R[key].append((rel, w))
    
    def query_concept(self, subjects, rel_filter=None, topn=5):
        """查询概念层关系"""
        results = []
        for subj in subjects:
            for (s, o), rels in self.concept_R.items():
                if s == subj:
                    for rel, w in rels:
                        if rel_filter is None or rel == rel_filter:
                            results.append((o, rel, w))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:topn]
    
    # ========== 场景层（中间层）==========
    def create_scene(self, scene_id, label, concept_ids):
        """创建场景：一组概念的组合"""
        self.scene_concepts[scene_id] = set(concept_ids)
        self.scene_labels[scene_id] = label
    
    def query_scene_by_concept(self, concept_id):
        """给定概念，找包含它的所有场景"""
        scenes = []
        for sid, cids in self.scene_concepts.items():
            if concept_id in cids:
                scenes.append(sid)
        return scenes
    
    def get_scene_concepts(self, scene_id):
        """获取场景包含的所有概念"""
        return self.scene_concepts.get(scene_id, set())
    
    # ========== 故事层（最高层）==========
    def create_story(self, story_id, label, scene_ids):
        """创建故事：一组场景的序列"""
        self.story_scenes[story_id] = list(scene_ids)
        self.story_labels[story_id] = label
    
    def query_story_by_scene(self, scene_id):
        """给定场景，找包含它的所有故事"""
        stories = []
        for sid, scids in self.story_scenes.items():
            if scene_id in scids:
                stories.append(sid)
        return stories
    
    # ========== 跨层传播（核心）==========
    def propagate(self, cue_concepts, max_depth=3):
        """
        跨层传播回忆：
        线索概念 → 所在场景 → 所在故事 → 故事的其他场景 → 其他场景的概念
        
        max_depth: 传播深度（防止无限扩散）
        """
        recalled_concepts = set(cue_concepts)
        recalled_scenes = set()
        recalled_stories = set()
        
        # 第1层：概念→场景
        for cid in cue_concepts:
            scenes = self.query_scene_by_concept(cid)
            recalled_scenes.update(scenes)
        
        # 第2层：场景→故事
        for sid in recalled_scenes:
            stories = self.query_story_by_scene(sid)
            recalled_stories.update(stories)
        
        # 第3层：故事→其他场景→其他概念
        for stid in recalled_stories:
            for sid in self.story_scenes.get(stid, []):
                if sid not in recalled_scenes:
                    # 新场景！
                    new_concepts = self.get_scene_concepts(sid)
                    recalled_concepts.update(new_concepts)
                    recalled_scenes.add(sid)
        
        return {
            "concepts": recalled_concepts,
            "scenes": recalled_scenes,
            "stories": recalled_stories,
        }
    
    def explain_recall(self, cue_concepts, result):
        """解释回忆路径（人类可读）"""
        paths = []
        for cid in cue_concepts:
            for sid in result["scenes"]:
                if cid in self.scene_concepts.get(sid, set()):
                    scene_label = self.scene_labels.get(sid, sid)
                    for stid in result["stories"]:
                        if sid in self.story_scenes.get(stid, []):
                            story_label = self.story_labels.get(stid, stid)
                            paths.append(f"{cid} → {scene_label} → {story_label}")
        return paths


if __name__ == "__main__":
    print("=" * 60)
    print("层次化R网测试")
    print("=" * 60)
    
    net = HierarchicalRNetwork()
    
    # 构建"乌鸦喝水"故事
    print("\n=== 构建故事：乌鸦喝水 ===")
    
    # 概念层
    net.add_concept_rel("乌鸦", "水", "需要")
    net.add_concept_rel("石子", "水面升高", "导致")
    net.add_concept_rel("水面升高", "喝水", "导致")
    net.add_concept_rel("渴", "喝水", "导致")
    print("  概念层关系: 乌鸦-需要-水, 石子-导致-水面升高, ...")
    
    # 场景层
    net.create_scene("scene_thirst", "乌鸦口渴", {"乌鸦", "渴"})
    net.create_scene("scene_stone", "放石子", {"乌鸦", "石子", "瓶子"})
    net.create_scene("scene_drink", "喝到水", {"乌鸦", "水", "喝水"})
    print("  场景层: 乌鸦口渴, 放石子, 喝到水")
    
    # 故事层
    net.create_story("story_crow", "乌鸦喝水", ["scene_thirst", "scene_stone", "scene_drink"])
    print("  故事层: 乌鸦喝水")
    
    # 测试：给线索"石子"，能想起整个故事
    print("\n=== 测试：线索'石子' → 跨层传播 ===")
    result = net.propagate({"石子"})
    print(f"  线索: 石子")
    print(f"  想起的场景: {[net.scene_labels.get(s, s) for s in result['scenes']]}")
    print(f"  想起的故事: {[net.story_labels.get(s, s) for s in result['stories']]}")
    print(f"  想起的全部概念: {result['concepts']}")
    
    assert "乌鸦" in result["concepts"], "应想起乌鸦"
    assert "水" in result["concepts"], "应想起水"
    assert "渴" in result["concepts"], "应想起渴"
    print("  [PASS] 跨层传播成功")
    
    # 测试：解释回忆路径
    print("\n=== 测试：回忆路径解释 ===")
    paths = net.explain_recall({"石子"}, result)
    for p in paths:
        print(f"  路径: {p}")
    
    print("\n" + "=" * 60)
    print("层次化R网测试全部通过 ✓")
    print("=" * 60)
