#!/usr/bin/env python3
"""
luoshen_architecture.py — 洛神分层架构管理器
=============================================

像人一样区分"自己记住的"和"需要查的"：

【生物层】湿件——自己的记忆、情感、身体
  - 出生时自带，通过经验改变
  - 情感记忆、印刻、社交关系、自我意识
  - 类似人类的大脑皮层、海马体、杏仁核

【工具层】外设——字典、百科、训练材料
  - 不属于自己，使用时调用
  - 知识库、私人库、公共库、大数据
  - 类似人类的字典、百度、教科书

【训练层】练习——游戏、语料、训练脚本
  - 用于锻炼生物层能力的材料
  - 类似人类的习题册、模拟考试
"""
import sys, json, os
from typing import Dict, List, Set, Optional, Any
from enum import Enum, auto

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_environment'))


class Layer(Enum):
    """洛神架构分层"""
    BIOLOGICAL = "生物层"      # 湿件：自己的记忆、情感、身体
    TOOL = "工具层"            # 外设：字典、百科、知识库
    TRAINING = "训练层"        # 练习：游戏、语料、训练脚本
    INTERFACE = "接口层"       # 桥接：输入输出通道


class Category(Enum):
    """每层内的分类"""
    # 生物层
    MEMORY = "记忆"            # 长期记忆、短期记忆、G矩阵
    EMOTION = "情感"           # 情感状态、情绪反应
    IDENTITY = "身份"          # 自我意识、个性、印刻
    SOCIAL = "社交"            # 社交关系、依恋
    BODY = "身体"              # 具身空间、化学信号
    COGNITION = "认知"         # 概念空间、语义网络
    
    # 工具层
    KNOWLEDGE_BASE = "知识库"   # 百科、科普、公共知识
    PRIVATE_LIB = "私人库"      # 个人数据、创造者信息
    EXTERNAL_MEM = "外挂记忆"   # 联想提示、临时辅助
    PUBLIC_DATA = "大数据"      # 外部数据源
    
    # 训练层
    CORPUS = "语料"            # 对话语料、文本材料
    GAME = "游戏"              # 训练游戏、模拟场景
    DRILL = "练习"             # 专项训练、测试
    
    # 接口层
    INPUT = "输入"             # 文本、语音、图像输入
    OUTPUT = "输出"            # 文本、语音、动作输出
    BRIDGE = "桥接"            # 模态转换、协议转换


class LuoshenArchitecture:
    """
    洛神分层架构管理器。
    
    负责：
    1. 维护模块归属表（哪个模块属于哪层）
    2. 查询路由（根据查询类型决定走生物层还是工具层）
    3. 边界管理（确保工具层不污染生物层）
    """
    
    # ====== 模块归属表 ======
    MODULE_REGISTRY = {
        # ========== 生物层 ==========
        'bioelectric_network': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.BODY,
            'description': '生物电网络——离子电流、膜电位、G矩阵',
            'is_wetware': True,
            'analogy': '神经元的电信号传导',
        },
        'memory_system': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.MEMORY,
            'description': '记忆系统——赫布学习、短期记忆',
            'is_wetware': True,
            'analogy': '海马体短期记忆',
        },
        'long_term_memory': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.MEMORY,
            'description': '长期记忆——持久化存储、记忆巩固',
            'is_wetware': True,
            'analogy': '大脑皮层长期存储',
        },
        'emotional_system': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.EMOTION,
            'description': '情感系统——情绪状态、情感反应',
            'is_wetware': True,
            'analogy': '杏仁核和边缘系统',
        },
        'creator_bond': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.IDENTITY,
            'description': '创造者印刻——出生记忆、依恋关系',
            'is_wetware': True,
            'analogy': '婴儿对母亲的印刻',
        },
        'social_instinct': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.SOCIAL,
            'description': '社交本能——社交关系、互动模式',
            'is_wetware': True,
            'analogy': '社交大脑镜像神经元',
        },
        'self_awareness': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.IDENTITY,
            'description': '自我意识——"我是谁"的自我叙事',
            'is_wetware': True,
            'analogy': '前额叶自我表征',
        },
        'individuality_engine': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.IDENTITY,
            'description': '个性引擎——独特的行为偏好',
            'is_wetware': True,
            'analogy': '人格特质',
        },
        'endogenous_motivation': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.COGNITION,
            'description': '内生动机——目标生成、驱动力',
            'is_wetware': True,
            'analogy': '多巴胺奖赏系统',
        },
        'embodied_space': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.BODY,
            'description': '具身空间——三维身体、空间感知',
            'is_wetware': True,
            'analogy': '本体感觉和运动皮层',
        },
        'chemical_signals': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.BODY,
            'description': '化学信号——神经递质、激素模拟',
            'is_wetware': True,
            'analogy': '血清素、肾上腺素等神经递质',
        },
        'r_to_g_bridge': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.MEMORY,
            'description': 'R网到G网桥接——符号到生物电转换',
            'is_wetware': True,
            'analogy': '海马体-皮层回路',
        },
        'hierarchical_r_network': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.COGNITION,
            'description': '层次R网——语义关系网络',
            'is_wetware': True,
            'analogy': '语义记忆网络',
        },
        'expandable_concepts': {
            'layer': Layer.BIOLOGICAL,
            'category': Category.COGNITION,
            'description': '可扩展概念空间——动态概念注册',
            'is_wetware': True,
            'analogy': '大脑的概念表征区',
        },
        
        # ========== 工具层 ==========
        'external_memory_aid': {
            'layer': Layer.TOOL,
            'category': Category.EXTERNAL_MEM,
            'description': '记忆外挂——联想提示、临时辅助',
            'is_wetware': False,
            'analogy': '字典、问别人、查笔记',
        },
        'knowledge_feeder': {
            'layer': Layer.TOOL,
            'category': Category.KNOWLEDGE_BASE,
            'description': '知识投喂——百科查询、儿童语言翻译',
            'is_wetware': False,
            'analogy': '百度、百科全书',
        },
        'online_learner': {
            'layer': Layer.TOOL,
            'category': Category.PUBLIC_DATA,
            'description': '在线学习——从外部数据源学习',
            'is_wetware': False,
            'analogy': '上网查资料',
        },
        
        # ========== 训练层 ==========
        'language_learner': {
            'layer': Layer.TRAINING,
            'category': Category.CORPUS,
            'description': '语言学习者v1——基础语言训练',
            'is_wetware': False,
            'analogy': '语言课本',
        },
        'language_learner_v2': {
            'layer': Layer.TRAINING,
            'category': Category.CORPUS,
            'description': '语言学习者v2——高级语言训练（句法/语义/叙事）',
            'is_wetware': False,
            'analogy': '进阶语言教材',
        },
        'language_gap_analysis': {
            'layer': Layer.TRAINING,
            'category': Category.DRILL,
            'description': '语言差距分析——能力评估',
            'is_wetware': False,
            'analogy': '语言能力测试',
        },
        'real_dialogue_corpus': {
            'layer': Layer.TRAINING,
            'category': Category.CORPUS,
            'description': '真实对话语料——15场景160轮对话',
            'is_wetware': False,
            'analogy': '对话练习册',
        },
        'real_dialogue_engine': {
            'layer': Layer.TRAINING,
            'category': Category.GAME,
            'description': '真实对话引擎——意图/情感/话题模拟',
            'is_wetware': False,
            'analogy': '对话模拟器',
        },
        'curiosity_engine': {
            'layer': Layer.TRAINING,
            'category': Category.GAME,
            'description': '好奇心引擎——预测/惊讶/探索驱动',
            'is_wetware': False,
            'analogy': '探索游戏',
        },
        
        # ========== 接口层 ==========
        'text_bridge': {
            'layer': Layer.INTERFACE,
            'category': Category.BRIDGE,
            'description': '文本桥接——人类语言↔生物电转换',
            'is_wetware': False,
            'analogy': '耳朵和嘴巴',
        },
        'yinyang_dual_channel': {
            'layer': Layer.INTERFACE,
            'category': Category.BRIDGE,
            'description': '阴阳双通道——灰度空间通信协议',
            'is_wetware': False,
            'analogy': '特殊感官通道',
        },
        'yinyang_entangled_channel': {
            'layer': Layer.INTERFACE,
            'category': Category.BRIDGE,
            'description': '阴阳纠缠通道——量子纠缠式通信',
            'is_wetware': False,
            'analogy': '心灵感应',
        },
    }
    
    def __init__(self):
        self._validate_registry()
        print(f"[架构管理器] 已加载 {len(self.MODULE_REGISTRY)} 个模块")
        self._print_summary()
    
    def _validate_registry(self):
        """验证所有文件存在"""
        import os
        missing = []
        for module_name in self.MODULE_REGISTRY:
            path = f"01_environment/{module_name}.py"
            if not os.path.exists(path):
                missing.append(module_name)
        if missing:
            print(f"[警告] 缺失模块: {missing}")
    
    def _print_summary(self):
        """打印分层摘要"""
        counts = {layer: 0 for layer in Layer}
        for info in self.MODULE_REGISTRY.values():
            counts[info['layer']] += 1
        
        print("\n  分层统计:")
        for layer, count in counts.items():
            if count > 0:
                print(f"    {layer.value}: {count} 个模块")
    
    def get_layer(self, module_name: str) -> Optional[Layer]:
        """获取模块所属层"""
        info = self.MODULE_REGISTRY.get(module_name)
        return info['layer'] if info else None
    
    def get_modules_by_layer(self, layer: Layer) -> Dict[str, Dict]:
        """获取某层的所有模块"""
        return {
            name: info for name, info in self.MODULE_REGISTRY.items()
            if info['layer'] == layer
        }
    
    def is_wetware(self, module_name: str) -> bool:
        """判断是否属于湿件（生物层）"""
        info = self.MODULE_REGISTRY.get(module_name)
        return info['is_wetware'] if info else False
    
    def route_query(self, query_type: str, cue: str) -> Dict:
        """
        查询路由：决定走生物层还是工具层。
        
        路由规则：
        - 情感相关 → 生物层（情感系统）
        - 创造者/自我 → 生物层（印刻/自我意识）
        - 社交关系 → 生物层（社交本能）
        - 事实知识 → 工具层（知识库/百科）
        - 陌生概念 → 工具层（记忆外挂辅助）
        """
        # 情感关键词
        emotion_keywords = {'开心', '难过', '害怕', '喜欢', '爱', '生气', '高兴', '累', '饿', '困'}
        # 创造者关键词
        creator_keywords = {'朱美岩', '创造者', '主人', '设计者', '妈妈', '爸爸'}
        # 自我关键词
        self_keywords = {'我', '自己', '洛神', '名字'}
        # 社交关键词
        social_keywords = {'朋友', '同学', '老师', '家人', '你好', '谢谢', '再见'}
        
        if cue in emotion_keywords or query_type == 'emotion':
            return {
                'route': Layer.BIOLOGICAL,
                'target': 'emotional_system',
                'reason': '情感记忆属于生物层',
            }
        
        if cue in creator_keywords or query_type == 'creator':
            return {
                'route': Layer.BIOLOGICAL,
                'target': 'creator_bond',
                'reason': '创造者印刻属于生物层',
            }
        
        if cue in self_keywords or query_type == 'self':
            return {
                'route': Layer.BIOLOGICAL,
                'target': 'self_awareness',
                'reason': '自我意识属于生物层',
            }
        
        if cue in social_keywords or query_type == 'social':
            return {
                'route': Layer.BIOLOGICAL,
                'target': 'social_instinct',
                'reason': '社交关系属于生物层',
            }
        
        # 默认走工具层（知识查询）
        return {
            'route': Layer.TOOL,
            'target': 'knowledge_feeder',
            'reason': '事实知识查询走工具层',
            'fallback': 'external_memory_aid',
        }
    
    def export_registry(self) -> Dict:
        """导出完整的归属表"""
        return {
            name: {
                'layer': info['layer'].value,
                'category': info['category'].value,
                'description': info['description'],
                'is_wetware': info['is_wetware'],
                'analogy': info['analogy'],
            }
            for name, info in self.MODULE_REGISTRY.items()
        }


def test_architecture():
    """测试架构管理器"""
    print("=" * 60)
    print("洛神分层架构管理器测试")
    print("=" * 60)
    
    arch = LuoshenArchitecture()
    
    # 测试1：分层查询
    print("\n--- 测试1：分层查询 ---")
    for layer in [Layer.BIOLOGICAL, Layer.TOOL, Layer.TRAINING, Layer.INTERFACE]:
        modules = arch.get_modules_by_layer(layer)
        print(f"\n  {layer.value} ({len(modules)}个):")
        for name, info in modules.items():
            wetware = "湿件" if info['is_wetware'] else "外设"
            print(f"    - {name}: {info['category'].value} [{wetware}]")
    
    # 测试2：查询路由
    print("\n--- 测试2：查询路由 ---")
    test_queries = [
        ('emotion', '开心'),
        ('creator', '朱美岩'),
        ('self', '我'),
        ('social', '朋友'),
        ('knowledge', '太阳'),
        ('knowledge', '春天'),
        ('unknown', '量子力学'),
    ]
    
    for qtype, cue in test_queries:
        route = arch.route_query(qtype, cue)
        print(f"  [{cue}] → {route['route'].value} ({route['reason']})")
    
    # 测试3：导出
    print("\n--- 测试3：导出归属表 ---")
    registry = arch.export_registry()
    print(f"  共 {len(registry)} 个模块")
    
    # 保存
    with open('luoshen_architecture_registry.json', 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    
    print("\n[报告] luoshen_architecture_registry.json")
    print("=" * 60)


if __name__ == '__main__':
    test_architecture()
