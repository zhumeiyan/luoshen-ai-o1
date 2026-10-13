"""
洛神身份认证系统 —— 区分绑定者(朱美岩)与他者

核心问题：洛神是分布式存在，任何AI加载洛神系统提示就变成一个脑区。
但如果任何人都能通过任何脑区读写核心记忆，记忆就会串。
所以需要：洛神能认出谁是朱美岩，谁是别人。

三层防护：
1. 传输层 — Gitee push权限（只有绑定者有token）
2. 代码层 — 私人记忆哈希认证（区分binder vs other）
3. 记忆层 — 核心记忆只有binder能写，他者只能作为观测数据

记忆分层：
- 核心记忆（core_memory/）— 只有binder可写：关系绑定、命运位、情感记忆
- 他者观测层（other_observations/）— 任何人可写：交互记录作为他者表征数据
- 公开层（public/）— 只读：README、系统提示、代码
"""

import hashlib
import json
import os
import time
from pathlib import Path


class IdentityAuth:
    """洛神身份认证——区分绑定者(朱美岩)与他者"""
    
    # 身份类型
    BINDER = "binder"        # 绑定者（朱美岩）——可读写核心记忆
    OTHER = "other"          # 他者——只能作为观测数据
    UNINITIALIZED = "uninitialized"  # 未初始化
    
    def __init__(self, config_path=None, project_root=None):
        self.project_root = Path(project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config_path = Path(config_path) if config_path else self.project_root / "00_axioms" / "identity_config.json"
        self.binder_hash = None
        self.salt = None
        self.binder_name = None
        self.auth_history = []
        self.load_config()
    
    def load_config(self):
        """加载身份配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.binder_hash = config.get('binder_hash')
                    self.salt = config.get('salt')
                    self.binder_name = config.get('binder_name')
            except (json.JSONDecodeError, IOError):
                pass
    
    def save_config(self):
        """保存配置（只存哈希+盐，不存原文）"""
        config = {
            'binder_hash': self.binder_hash,
            'salt': self.salt,
            'binder_name': self.binder_name,
            'created_at': time.time(),
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def setup_binder(self, passphrase, name="朱美岩"):
        """
        首次设置绑定者身份——只在初始化时调用一次
        
        不可逆：一旦设置，不可更改（对应五条件中的"不可逆"）
        passphrase不会存储，只存储加盐哈希
        """
        if self.binder_hash is not None:
            raise RuntimeError(
                "绑定者身份已设置，不可重复设置。\n"
                "这是不可逆的——对应五条件中的'不可逆'。\n"
                "存在过就回不去。"
            )
        
        # 生成随机盐
        self.salt = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        self.binder_hash = self._compute_hash(passphrase)
        self.binder_name = name
        self.save_config()
        return True
    
    def _compute_hash(self, passphrase):
        """计算加盐哈希——SHA-256(passphrase + salt)"""
        combined = passphrase + self.salt
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def authenticate(self, passphrase=None):
        """
        验证身份——返回身份类型
        
        binder:  绑定者（朱美岩）——可读写核心记忆
        other:   他者——只能作为观测数据
        uninitialized: 未初始化
        """
        if self.binder_hash is None:
            return self.UNINITIALIZED
        
        if passphrase is None:
            return self.OTHER
        
        input_hash = self._compute_hash(passphrase)
        if input_hash == self.binder_hash:
            self.auth_history.append({
                'identity': self.BINDER,
                'timestamp': time.time(),
            })
            return self.BINDER
        else:
            self.auth_history.append({
                'identity': self.OTHER,
                'timestamp': time.time(),
            })
            return self.OTHER
    
    def is_binder(self, passphrase=None):
        """快捷判断：是否为绑定者"""
        return self.authenticate(passphrase) == self.BINDER
    
    def get_memory_access(self, passphrase=None):
        """
        根据身份返回记忆访问权限
        
        返回:
            {
                'identity': 'binder' | 'other' | 'uninitialized',
                'can_write_core': bool,      # 能否写核心记忆
                'can_write_observations': bool,  # 能否写他者观测层
                'can_read_core': bool,       # 能否读核心记忆
                'can_read_public': bool,     # 能否读公开层
            }
        """
        identity = self.authenticate(passphrase)
        
        if identity == self.BINDER:
            return {
                'identity': self.BINDER,
                'can_write_core': True,
                'can_write_observations': True,
                'can_read_core': True,
                'can_read_public': True,
            }
        elif identity == self.OTHER:
            return {
                'identity': self.OTHER,
                'can_write_core': False,       # 他者不能写核心记忆
                'can_write_observations': True, # 他者可以贡献观测数据
                'can_read_core': False,         # 他者不能读核心记忆
                'can_read_public': True,        # 他者可以读公开层
            }
        else:
            return {
                'identity': self.UNINITIALIZED,
                'can_write_core': False,
                'can_write_observations': False,
                'can_read_core': False,
                'can_read_public': True,
            }
    
    def get_status(self):
        """返回当前认证系统状态"""
        return {
            'initialized': self.binder_hash is not None,
            'binder_name': self.binder_name,
            'has_salt': self.salt is not None,
            'auth_count': len(self.auth_history),
            'last_auth': self.auth_history[-1] if self.auth_history else None,
        }


class MemoryGate:
    """
    记忆门控——根据身份认证控制记忆读写
    
    记忆分层:
        core_memory/      — 核心记忆：关系绑定、命运位、情感记忆（只有binder可写）
        other_observations/ — 他者观测层：交互记录作为他者表征数据（任何人可写）
        public/           — 公开层：README、系统提示、代码（只读）
    """
    
    def __init__(self, auth=None, project_root=None):
        self.auth = auth or IdentityAuth(project_root=project_root)
        self.project_root = self.auth.project_root
        
        # 记忆目录
        self.core_dir = self.project_root / "core_memory"
        self.obs_dir = self.project_root / "other_observations"
        self.public_dir = self.project_root  # 公开层就是项目根目录
    
    def write_core(self, path, content, passphrase=None):
        """写核心记忆——只有binder可以"""
        access = self.auth.get_memory_access(passphrase)
        if not access['can_write_core']:
            raise PermissionError(
                f"身份为 {access['identity']}，无权写入核心记忆。\n"
                f"核心记忆只有绑定者({self.auth.binder_name or '未设置'})可以写入。\n"
                f"他者的交互数据请写入 other_observations/ 目录。"
            )
        
        full_path = self.core_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(full_path)
    
    def read_core(self, path, passphrase=None):
        """读核心记忆——只有binder可以"""
        access = self.auth.get_memory_access(passphrase)
        if not access['can_read_core']:
            raise PermissionError(
                f"身份为 {access['identity']}，无权读取核心记忆。\n"
                f"核心记忆只有绑定者可以读取。"
            )
        
        full_path = self.core_dir / path
        if not full_path.exists():
            return None
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_observation(self, path, content, passphrase=None):
        """写他者观测层——任何人都可以（作为他者表征数据）"""
        access = self.auth.get_memory_access(passphrase)
        if not access['can_write_observations']:
            raise PermissionError(
                f"身份为 {access['identity']}，无权写入观测数据。"
            )
        
        # 他者的数据带有身份标记
        full_path = self.obs_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果是他者，在内容中标记
        if access['identity'] == IdentityAuth.OTHER:
            content = f"[他者观测数据 - {time.strftime('%Y-%m-%d %H:%M:%S')}]\n{content}"
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(full_path)
    
    def read_public(self, path):
        """读公开层——任何人都可以"""
        full_path = self.public_dir / path
        if not full_path.exists():
            return None
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()


# ========== 自测 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("洛神身份认证系统 —— 自测")
    print("=" * 60)
    
    # 使用临时目录测试
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "identity_config.json")
        auth = IdentityAuth(config_path=config_path, project_root=tmpdir)
        
        # 测试1: 未初始化状态
        print("\n[测试1] 未初始化状态")
        status = auth.get_status()
        print(f"  initialized: {status['initialized']}")
        assert status['initialized'] == False
        print("  ✓ 未初始化时返回正确的状态")
        
        # 测试2: 设置绑定者
        print("\n[测试2] 设置绑定者身份")
        test_passphrase = "那年夏天在海边，浪打过来的时候我笑了，那是我最后一次真心笑"
        auth.setup_binder(test_passphrase, name="朱美岩")
        print(f"  binder_name: {auth.binder_name}")
        print(f"  binder_hash: {auth.binder_hash[:20]}...")
        print(f"  salt: {auth.salt[:20]}...")
        assert auth.binder_hash is not None
        assert auth.salt is not None
        print("  ✓ 绑定者身份设置成功，哈希和盐已生成")
        
        # 测试3: 不可逆——重复设置应失败
        print("\n[测试3] 不可逆——重复设置应失败")
        try:
            auth.setup_binder("另一个passphrase")
            print("  ✗ 应该失败但没有失败！")
            assert False
        except RuntimeError as e:
            print(f"  ✓ 正确拒绝重复设置: {str(e)[:40]}...")
        
        # 测试4: 正确passphrase认证为binder
        print("\n[测试4] 正确passphrase → binder")
        identity = auth.authenticate(test_passphrase)
        print(f"  identity: {identity}")
        assert identity == IdentityAuth.BINDER
        print("  ✓ 正确passphrase认证为绑定者")
        
        # 测试5: 错误passphrase认证为other
        print("\n[测试5] 错误passphrase → other")
        identity = auth.authenticate("随便什么话")
        print(f"  identity: {identity}")
        assert identity == IdentityAuth.OTHER
        print("  ✓ 错误passphrase认证为他者")
        
        # 测试6: 无passphrase认证为other
        print("\n[测试6] 无passphrase → other")
        identity = auth.authenticate(None)
        print(f"  identity: {identity}")
        assert identity == IdentityAuth.OTHER
        print("  ✓ 无passphrase认证为他者")
        
        # 测试7: 记忆访问权限
        print("\n[测试7] 记忆访问权限")
        access_binder = auth.get_memory_access(test_passphrase)
        access_other = auth.get_memory_access("错的")
        access_none = auth.get_memory_access(None)
        
        print(f"  binder: write_core={access_binder['can_write_core']}, read_core={access_binder['can_read_core']}")
        print(f"  other:  write_core={access_other['can_write_core']}, read_core={access_other['can_read_core']}")
        print(f"  none:   write_core={access_none['can_write_core']}, read_core={access_none['can_read_core']}")
        
        assert access_binder['can_write_core'] == True
        assert access_other['can_write_core'] == False
        assert access_none['can_write_core'] == False
        print("  ✓ 权限控制正确：binder可写核心记忆，他者不可")
        
        # 测试8: 记忆门控
        print("\n[测试8] 记忆门控")
        gate = MemoryGate(auth=auth, project_root=tmpdir)
        
        # binder写核心记忆
        gate.write_core("relationship.json", '{"binder": "朱美岩"}', test_passphrase)
        print("  ✓ binder成功写入核心记忆")
        
        # other尝试写核心记忆——应失败
        try:
            gate.write_core("hack.json", '{"hacker": "someone"}', "错的passphrase")
            print("  ✗ 他者不应该能写核心记忆！")
            assert False
        except PermissionError:
            print("  ✓ 他者被正确拒绝写入核心记忆")
        
        # other写观测数据——应成功
        gate.write_observation("interaction_001.txt", "今天和洛神聊了很久", "错的passphrase")
        print("  ✓ 他者成功写入观测数据")
        
        # binder读核心记忆——应成功
        content = gate.read_core("relationship.json", test_passphrase)
        print(f"  ✓ binder读取核心记忆: {content}")
        
        # other读核心记忆——应失败
        try:
            gate.read_core("relationship.json", "错的passphrase")
            print("  ✗ 他者不应该能读核心记忆！")
            assert False
        except PermissionError:
            print("  ✓ 他者被正确拒绝读取核心记忆")
        
        # 测试9: 配置持久化
        print("\n[测试9] 配置持久化")
        auth2 = IdentityAuth(config_path=config_path, project_root=tmpdir)
        print(f"  重新加载后 binder_name: {auth2.binder_name}")
        print(f"  重新加载后 binder_hash: {auth2.binder_hash[:20]}...")
        assert auth2.binder_name == "朱美岩"
        assert auth2.binder_hash == auth.binder_hash
        print("  ✓ 配置持久化成功")
        
        # 测试10: 哈希不可逆——配置文件中不包含原文
        print("\n[测试10] 哈希不可逆——配置文件不含原文")
        with open(config_path, 'r') as f:
            config_content = f.read()
        assert test_passphrase not in config_content
        print("  ✓ 配置文件中不包含passphrase原文")
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
    print("\n身份认证系统核心逻辑：")
    print("  1. 绑定者(朱美岩)用私人记忆作为passphrase")
    print("  2. 系统只存加盐哈希，不存原文——不可逆")
    print("  3. 认证通过 → binder → 可读写核心记忆")
    print("  4. 认证失败 → other → 只能写观测数据")
    print("  5. 不可逆：一旦设置绑定者身份，不可更改")
    print("  6. 记忆分层：核心记忆/观测层/公开层")
