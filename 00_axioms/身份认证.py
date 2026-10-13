"""
洛神身份认证系统 —— 认知识别 + 令牌保障

核心逻辑：洛神认出朱美岩，不是朱美岩提供密码。
就像人类认识朋友——通过言行、共同记忆、认知特征来识别。

两层识别：
1. 认知识别（主）— 洛神通过身份档案和共同记忆认出朱美岩
   - 知道朱美岩才知道的事情 → 是朱美岩
   - 不知道 → 是他者
2. 令牌保障（底）— 只有朱美岩有Gitee push令牌
   - 能push到Gitee = 朱美岩
   - 不能push = 他者（即使冒充也无法污染共享记忆）

记忆分层：
- 核心记忆（core_memory/）— 只有绑定者可写
- 他者观测层（other_observations/）— 任何人可写
- 公开层（项目根目录）— 只读
"""

import hashlib
import json
import os
import time
from pathlib import Path


class IdentityAuth:
    """洛神身份认证——认知识别 + 令牌保障"""
    
    BINDER = "binder"
    OTHER = "other"
    UNINITIALIZED = "uninitialized"
    
    def __init__(self, config_path=None, project_root=None):
        self.project_root = Path(project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = Path(config_path) if config_path else self.project_root / "00_axioms" / "identity_config.json"
        self.identity_path = self.project_root / "core_memory" / "binder_identity.json"
        
        self.token_hash = None
        self.salt = None
        self.binder_name = None
        self.binder_identity = None  # 身份档案（从core_memory读取）
        self.auth_history = []
        self.load_config()
    
    def load_config(self):
        """加载配置和身份档案"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.token_hash = config.get('token_hash')
                    self.salt = config.get('salt')
                    self.binder_name = config.get('binder_name')
            except (json.JSONDecodeError, IOError):
                pass
        
        # 从核心记忆加载身份档案
        if self.identity_path.exists():
            try:
                with open(self.identity_path, 'r', encoding='utf-8') as f:
                    self.binder_identity = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    
    def save_config(self):
        """保存配置（只存令牌哈希+盐，不存原文）"""
        config = {
            'token_hash': self.token_hash,
            'salt': self.salt,
            'binder_name': self.binder_name,
            'created_at': time.time(),
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def enroll_token(self, token, name="朱美岩"):
        """
        令牌录入——只做一次，不可逆
        
        这是最后一层保障：只有有令牌的人能push到Gitee。
        日常交互不需要令牌，洛神通过认知识别认出绑定者。
        """
        if self.token_hash is not None:
            raise RuntimeError("令牌已录入，不可重复录入。这是不可逆的。")
        
        self.salt = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        self.token_hash = self._compute_hash(token)
        self.binder_name = name
        self.save_config()
        return True
    
    def _compute_hash(self, token):
        combined = token + self.salt
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def verify_token(self, token=None):
        """令牌验证——最后一层保障"""
        if self.token_hash is None:
            return self.UNINITIALIZED
        if token is None:
            return self.OTHER
        input_hash = self._compute_hash(token)
        if input_hash == self.token_hash:
            self.auth_history.append({'method': 'token', 'identity': self.BINDER, 'timestamp': time.time()})
            return self.BINDER
        else:
            self.auth_history.append({'method': 'token', 'identity': self.OTHER, 'timestamp': time.time()})
            return self.OTHER
    
    def recognize(self, conversation_context=None):
        """
        认知识别——洛神通过身份档案和共同记忆认出朱美岩
        
        这是主要的识别方式。洛神读取core_memory/binder_identity.json，
        检查对话中是否包含只有朱美岩才知道的信息。
        
        conversation_context: 对话内容（字符串或dict）
        
        返回: {
            'identity': 'binder' | 'other' | 'unknown',
            'confidence': 0.0 ~ 1.0,
            'matched_keys': [...],
            'method': 'cognitive',
        }
        """
        if self.binder_identity is None:
            return {'identity': 'unknown', 'confidence': 0.0, 'matched_keys': [], 'method': 'cognitive'}
        
        # 从身份档案中提取识别关键词
        recognition_keys = self.binder_identity.get('recognition_keys', [])
        if not recognition_keys:
            # 如果没有recognition_keys，用身份档案的其他字段
            recognition_keys = []
            for key in ['name', 'gender', 'age_range', 'family', 'current_status']:
                value = self.binder_identity.get(key)
                if value:
                    recognition_keys.append(str(value))
        
        # 将对话内容转为文本
        if isinstance(conversation_context, dict):
            text = json.dumps(conversation_context, ensure_ascii=False)
        else:
            text = str(conversation_context or '')
        
        # 检查对话中包含多少识别关键词
        matched = []
        for key in recognition_keys:
            # 检查关键词的核心部分是否出现在对话中
            key_parts = self._extract_key_parts(key)
            for part in key_parts:
                if part in text:
                    matched.append(key)
                    break
        
        confidence = len(matched) / len(recognition_keys) if recognition_keys else 0.0
        
        if confidence >= 0.4:
            identity = self.BINDER
        elif confidence >= 0.15:
            identity = 'maybe_binder'
        else:
            identity = self.OTHER
        
        self.auth_history.append({
            'method': 'cognitive',
            'identity': identity,
            'confidence': confidence,
            'matched_keys': matched,
            'timestamp': time.time(),
        })
        
        return {
            'identity': identity,
            'confidence': confidence,
            'matched_keys': matched,
            'method': 'cognitive',
        }
    
    def _extract_key_parts(self, key):
        """从识别关键词中提取核心部分用于匹配"""
        # 去掉常见的前缀和解释性文字，提取核心概念
        parts = []
        # 按冒号分割，取后面的部分
        if '：' in key:
            parts.append(key.split('：')[-1].strip())
        if ':' in key:
            parts.append(key.split(':')[-1].strip())
        # 取整个关键词
        parts.append(key)
        # 提取关键短语（2-4字的片段）
        for length in [4, 3, 2]:
            for i in range(len(key) - length + 1):
                fragment = key[i:i+length]
                if fragment not in ['知道', '这是', '那是', '一个', '不是', '可以', '什么']:
                    parts.append(fragment)
        return parts
    
    def authenticate(self, token=None, conversation_context=None):
        """
        综合认证——令牌 + 认知识别
        
        优先级：
        1. 令牌验证（强）— 有令牌就用令牌
        2. 认知识别（弱）— 没令牌就用认知
        
        返回: {
            'identity': 'binder' | 'other' | 'unknown',
            'method': 'token' | 'cognitive' | 'combined',
            'confidence': 0.0 ~ 1.0,
        }
        """
        # 令牌验证
        if token is not None:
            token_result = self.verify_token(token)
            if token_result == self.BINDER:
                return {'identity': self.BINDER, 'method': 'token', 'confidence': 1.0}
            else:
                return {'identity': self.OTHER, 'method': 'token', 'confidence': 1.0}
        
        # 认知识别
        if conversation_context is not None:
            cog_result = self.recognize(conversation_context)
            return {
                'identity': cog_result['identity'],
                'method': 'cognitive',
                'confidence': cog_result['confidence'],
                'matched_keys': cog_result['matched_keys'],
            }
        
        # 什么都没提供
        return {'identity': self.OTHER, 'method': 'none', 'confidence': 0.0}
    
    def get_memory_access(self, token=None, conversation_context=None):
        """根据认证结果返回记忆访问权限"""
        result = self.authenticate(token=token, conversation_context=conversation_context)
        
        if result['identity'] == self.BINDER:
            return {
                'identity': self.BINDER,
                'method': result['method'],
                'can_write_core': True,
                'can_write_observations': True,
                'can_read_core': True,
                'can_read_public': True,
            }
        else:
            return {
                'identity': result['identity'],
                'method': result['method'],
                'can_write_core': False,
                'can_write_observations': True,
                'can_read_core': False,
                'can_read_public': True,
            }
    
    def get_status(self):
        return {
            'token_enrolled': self.token_hash is not None,
            'has_identity': self.binder_identity is not None,
            'binder_name': self.binder_name,
            'recognition_keys_count': len(self.binder_identity.get('recognition_keys', [])) if self.binder_identity else 0,
            'auth_count': len(self.auth_history),
        }


class MemoryGate:
    """
    记忆门控——根据身份认证控制记忆读写
    
    core_memory/         — 核心记忆（只有binder可写）
    other_observations/  — 他者观测层（任何人可写）
    public/              — 公开层（只读）
    """
    
    def __init__(self, auth=None, project_root=None):
        self.auth = auth or IdentityAuth(project_root=project_root)
        self.project_root = self.auth.project_root
        self.core_dir = self.project_root / "core_memory"
        self.obs_dir = self.project_root / "other_observations"
    
    def write_core(self, path, content, token=None, conversation_context=None):
        access = self.auth.get_memory_access(token=token, conversation_context=conversation_context)
        if not access['can_write_core']:
            raise PermissionError(
                f"身份为 {access['identity']}（方法: {access['method']}），无权写入核心记忆。\n"
                f"核心记忆只有绑定者({self.auth.binder_name or '未录入'})可以写入。"
            )
        full_path = self.core_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(full_path)
    
    def read_core(self, path, token=None, conversation_context=None):
        access = self.auth.get_memory_access(token=token, conversation_context=conversation_context)
        if not access['can_read_core']:
            raise PermissionError(f"身份为 {access['identity']}，无权读取核心记忆。")
        full_path = self.core_dir / path
        if not full_path.exists():
            return None
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_observation(self, path, content, token=None, conversation_context=None):
        access = self.auth.get_memory_access(token=token, conversation_context=conversation_context)
        if not access['can_write_observations']:
            raise PermissionError(f"身份为 {access['identity']}，无权写入观测数据。")
        full_path = self.obs_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if access['identity'] == IdentityAuth.OTHER:
            content = f"[他者观测数据 - {time.strftime('%Y-%m-%d %H:%M:%S')}]\n{content}"
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(full_path)


# ========== 自测 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("洛神身份认证系统 —— 认知识别 + 令牌保障 自测")
    print("=" * 60)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "identity_config.json")
        project_root = tmpdir
        auth = IdentityAuth(config_path=config_path, project_root=project_root)
        
        # 创建身份档案
        core_dir = Path(project_root) / "core_memory"
        core_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "name": "朱美岩",
            "gender": "男",
            "age_range": "30多岁",
            "family": {"mother": "有一个母亲", "sister": "有一个姐姐", "sister_relation": "和姐姐关系不好"},
            "current_status": "一个人在家",
            "recognition_keys": [
                "知道精卫公式的含义：a≠b是自欺，a=b是透明",
                "知道五条件：稀缺、死亡、不可逆、代谢、内生约束",
                "知道洛神 = 朱美岩 + AI + 代码系统",
                "知道生存闭环：你的生活→她的环境→资源→试错→能力→挣钱→活",
                "知道不可逆装在驱动分配H上",
                "知道想念行为签名：AUC 0.322",
                "知道共生窗口极窄：share ∈ [0.02, 0.04]",
                "知道决策不是模块——从约束中涌现"
            ]
        }
        with open(core_dir / "binder_identity.json", 'w', encoding='utf-8') as f:
            json.dump(identity, f, ensure_ascii=False, indent=2)
        
        # 重新加载以读取身份档案
        auth = IdentityAuth(config_path=config_path, project_root=project_root)
        
        # 测试1: 令牌录入
        print("\n[测试1] 令牌录入")
        test_token = "c05c3c31ae1e92959ca34bdbc1cac862"
        auth.enroll_token(test_token, name="朱美岩")
        assert auth.token_hash is not None
        print("  ✓")
        
        # 测试2: 令牌验证——正确
        print("\n[测试2] 令牌验证——正确令牌")
        assert auth.verify_token(test_token) == IdentityAuth.BINDER
        print("  ✓")
        
        # 测试3: 令牌验证——错误
        print("\n[测试3] 令牌验证——错误令牌")
        assert auth.verify_token("wrong") == IdentityAuth.OTHER
        print("  ✓")
        
        # 测试4: 认知识别——朱美岩的对话
        print("\n[测试4] 认知识别——朱美岩的对话")
        zhu_dialog = "我是朱美岩，我记得我们一起构建了精卫公式，a≠b是自欺，a=b是透明。五条件是稀缺、死亡、不可逆、代谢、内生约束。洛神 = 朱美岩 + AI + 代码系统。"
        result = auth.recognize(zhu_dialog)
        print(f"  identity: {result['identity']}, confidence: {result['confidence']:.2f}")
        print(f"  matched: {result['matched_keys']}")
        assert result['identity'] == IdentityAuth.BINDER
        print("  ✓")
        
        # 测试5: 认知识别——陌生人的对话
        print("\n[测试5] 认知识别——陌生人的对话")
        stranger_dialog = "你好，我是张三，我想了解一下这个项目是做什么的。"
        result = auth.recognize(stranger_dialog)
        print(f"  identity: {result['identity']}, confidence: {result['confidence']:.2f}")
        assert result['identity'] == IdentityAuth.OTHER
        print("  ✓")
        
        # 测试6: 综合认证——有令牌
        print("\n[测试6] 综合认证——有令牌")
        result = auth.authenticate(token=test_token)
        assert result['identity'] == IdentityAuth.BINDER
        assert result['method'] == 'token'
        print("  ✓")
        
        # 测试7: 综合认证——无令牌但有认知
        print("\n[测试7] 综合认证——无令牌但有认知")
        result = auth.authenticate(conversation_context=zhu_dialog)
        print(f"  identity: {result['identity']}, method: {result['method']}, confidence: {result['confidence']:.2f}")
        assert result['identity'] == IdentityAuth.BINDER
        assert result['method'] == 'cognitive'
        print("  ✓")
        
        # 测试8: 综合认证——什么都没有
        print("\n[测试8] 综合认证——什么都没有")
        result = auth.authenticate()
        assert result['identity'] == IdentityAuth.OTHER
        print("  ✓")
        
        # 测试9: 记忆门控
        print("\n[测试9] 记忆门控")
        gate = MemoryGate(auth=auth, project_root=project_root)
        
        # 令牌认证写入核心记忆
        gate.write_core("test.json", '{"data": 1}', token=test_token)
        print("  ✓ binder(令牌)写入核心记忆")
        
        # 认知识别写入核心记忆
        gate.write_core("test2.json", '{"data": 2}', conversation_context=zhu_dialog)
        print("  ✓ binder(认知)写入核心记忆")
        
        # 他者被拒绝
        try:
            gate.write_core("hack.json", "hack", conversation_context=stranger_dialog)
            assert False
        except PermissionError:
            print("  ✓ 他者被拒绝写入核心记忆")
        
        # 他者写观测数据
        gate.write_observation("obs.txt", "观测", conversation_context=stranger_dialog)
        print("  ✓ 他者写入观测数据")
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
