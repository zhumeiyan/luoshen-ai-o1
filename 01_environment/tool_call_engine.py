#!/usr/bin/env python3
"""
tool_call_engine.py — AI工具调用引擎
=====================================

核心理念：洛神像人一样使用AI工具。

使用场景：
- 洛神不会画画 → 调用文生图API
- 洛神不会配音 → 调用语音合成
- 洛神不会写代码 → 调用大模型生成代码
- 洛神想不起知识 → 调用搜索引擎

不是替代洛神的能力，是扩展她的能力边界。
类比：人类不会飞，但会用飞机；不会算复杂公式，但会用计算器。

设计原则：
1. 工具是"外设"，不是洛神本身
2. 洛神需要知道"什么时候该用什么工具"
3. 工具调用记录会存入记忆，供后续学习
4. 最终目标：洛神学会工具原理，自己实现
"""

import json, os, subprocess, tempfile, textwrap
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolCall:
    """一次工具调用的记录"""
    tool_name: str
    params: Dict[str, Any]
    result: Any = None
    success: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


class BaseTool:
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def call(self, **params) -> Dict:
        """执行工具调用，子类必须实现"""
        raise NotImplementedError
    
    def get_schema(self) -> Dict:
        """返回工具的参数schema，供LLM理解"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {}
        }


class LLMChatTool(BaseTool):
    """
    大模型对话工具——洛神的"智囊团"。
    
    当洛神遇到自己解决不了的问题时，可以"请教"大模型。
    不是替代洛神的思考，是辅助。
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'default'):
        super().__init__('llm_chat', '大模型对话——通用问答和文本生成')
        self.api_key = api_key or os.getenv('LLM_API_KEY')
        self.model = model
    
    def call(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 500) -> Dict:
        """
        调用大模型生成回答。
        
        Args:
            prompt: 用户问题
            system_prompt: 系统提示词
            max_tokens: 最大生成长度
        """
        if not self.api_key:
            return {
                'success': False,
                'result': None,
                'error': 'API密钥未配置，请在环境变量中设置 LLM_API_KEY'
            }
        
        # 这里接入实际的大模型API
        # 目前先用模拟实现，展示接口
        try:
            # 模拟调用（实际使用时替换为真实API调用）
            simulated_response = self._simulate_response(prompt)
            return {
                'success': True,
                'result': simulated_response,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def _simulate_response(self, prompt: str) -> str:
        """模拟大模型响应（实际使用时删除）"""
        # 简单规则匹配，模拟智能
        if '写' in prompt or '代码' in prompt:
            return "我可以帮你写代码。请告诉我具体需求。"
        if '画' in prompt or '图' in prompt:
            return "我可以生成图像。请描述你想要的画面。"
        if '为什么' in prompt:
            return "这是一个好问题。让我思考一下..."
        return f"我理解你的问题：'{prompt[:50]}...'。让我想想怎么回答。"
    
    def get_schema(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'prompt': {
                    'type': 'string',
                    'description': '要询问大模型的问题或任务',
                    'required': True
                },
                'system_prompt': {
                    'type': 'string',
                    'description': '系统提示词，定义大模型的角色',
                    'required': False
                },
                'max_tokens': {
                    'type': 'integer',
                    'description': '最大生成token数',
                    'default': 500
                }
            }
        }


class CodeExecutorTool(BaseTool):
    """
    代码执行工具——洛神的"双手"。
    
    洛神可以生成Python代码并执行，操作文件、处理数据等。
    安全原则：在沙箱环境中运行，限制资源使用。
    """
    
    def __init__(self, timeout: int = 30):
        super().__init__('code_execute', '代码执行——运行Python代码')
        self.timeout = timeout
    
    def call(self, code: str, context: Optional[Dict] = None) -> Dict:
        """
        执行Python代码。
        
        Args:
            code: Python代码字符串
            context: 执行上下文变量
        """
        # 安全检查：禁止危险操作
        dangerous = ['os.system', 'subprocess', 'eval(', 'exec(', '__import__', "open('/"]
        for d in dangerous:
            if d in code:
                return {
                    'success': False,
                    'result': None,
                    'error': f'安全拦截：代码包含危险操作 "{d}"'
                }
        
        try:
            # 创建临时文件执行
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # 清理
            os.unlink(temp_file)
            
            return {
                'success': result.returncode == 0,
                'result': {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                },
                'error': None if result.returncode == 0 else result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'result': None,
                'error': f'代码执行超时（>{self.timeout}秒）'
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def get_schema(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'code': {
                    'type': 'string',
                    'description': '要执行的Python代码',
                    'required': True
                },
                'context': {
                    'type': 'object',
                    'description': '执行上下文变量',
                    'required': False
                }
            }
        }


class TextToSpeechTool(BaseTool):
    """
    语音合成工具——洛神的"嘴巴"。
    
    把文本转换为语音，让洛神能"说话"。
    """
    
    def __init__(self, voice: str = 'default'):
        super().__init__('text_to_speech', '语音合成——文本转语音')
        self.voice = voice
    
    def call(self, text: str, output_file: Optional[str] = None) -> Dict:
        """
        将文本转换为语音。
        
        Args:
            text: 要转换的文本
            output_file: 输出文件路径（可选）
        """
        try:
            # 这里可以接入百度语音合成或其他TTS服务
            # 目前返回文本表示已"说话"
            if not output_file:
                output_file = f"tts_output_{datetime.now().strftime('%H%M%S')}.mp3"
            
            return {
                'success': True,
                'result': {
                    'text': text,
                    'output_file': output_file,
                    'voice': self.voice,
                    'status': '已生成语音（模拟）'
                },
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def get_schema(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'text': {
                    'type': 'string',
                    'description': '要转换为语音的文本',
                    'required': True
                },
                'output_file': {
                    'type': 'string',
                    'description': '输出音频文件路径',
                    'required': False
                }
            }
        }


class ImageGenerationTool(BaseTool):
    """
    图像生成工具——洛神的"画笔"。
    
    根据描述生成图像，让洛神能"画画"。
    """
    
    def __init__(self, size: str = '1024x1024'):
        super().__init__('image_generate', '图像生成——文生图')
        self.size = size
    
    def call(self, prompt: str, style: Optional[str] = None) -> Dict:
        """
        根据描述生成图像。
        
        Args:
            prompt: 图像描述
            style: 风格（如"卡通"/"写实"/"油画"）
        """
        try:
            # 这里可以接入百度图像生成或其他文生图服务
            return {
                'success': True,
                'result': {
                    'prompt': prompt,
                    'style': style or 'default',
                    'size': self.size,
                    'status': '已生成图像（模拟）',
                    'note': '实际使用时接入baidu-image-gen技能'
                },
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def get_schema(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'prompt': {
                    'type': 'string',
                    'description': '图像内容描述',
                    'required': True
                },
                'style': {
                    'type': 'string',
                    'description': '图像风格',
                    'required': False
                }
            }
        }


class SearchTool(BaseTool):
    """
    搜索工具——洛神的"望远镜"。
    
    搜索互联网获取实时信息。
    """
    
    def __init__(self):
        super().__init__('web_search', '网络搜索——获取实时信息')
    
    def call(self, query: str, num_results: int = 5) -> Dict:
        """
        搜索网络信息。
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量
        """
        try:
            # 这里可以接入websearch工具
            # 模拟返回结果
            return {
                'success': True,
                'result': {
                    'query': query,
                    'num_results': num_results,
                    'results': [
                        {'title': f'关于"{query}"的结果1', 'url': 'https://example.com/1'},
                        {'title': f'关于"{query}"的结果2', 'url': 'https://example.com/2'},
                    ],
                    'note': '实际使用时接入websearch工具'
                },
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def get_schema(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'query': {
                    'type': 'string',
                    'description': '搜索关键词',
                    'required': True
                },
                'num_results': {
                    'type': 'integer',
                    'description': '返回结果数量',
                    'default': 5
                }
            }
        }


class ToolCallEngine:
    """
    AI工具调用引擎——洛神的"工具箱"。
    
    核心功能：
    1. 工具注册和管理
    2. 智能路由（根据需求自动选择工具）
    3. 调用记录（存入记忆供学习）
    4. 结果解析（把工具输出转化为洛神能理解的形式）
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.call_history: List[ToolCall] = []
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认工具"""
        self.register_tool(LLMChatTool())
        self.register_tool(CodeExecutorTool())
        self.register_tool(TextToSpeechTool())
        self.register_tool(ImageGenerationTool())
        self.register_tool(SearchTool())
    
    def register_tool(self, tool: BaseTool):
        """注册新工具"""
        self.tools[tool.name] = tool
        print(f"[工具注册] {tool.name}: {tool.description}")
    
    def list_tools(self) -> List[Dict]:
        """列出所有可用工具"""
        return [tool.get_schema() for tool in self.tools.values()]
    
    def route_tool(self, need: str) -> Optional[str]:
        """
        智能路由：根据需求选择最合适的工具。
        
        路由规则（简单版，后续可升级为大模型决策）：
        - 包含"写代码"/"编程"/"Python" → code_execute
        - 包含"画"/"图"/"生成图像" → image_generate
        - 包含"说"/"读"/"语音" → text_to_speech
        - 包含"搜索"/"查"/"找" → web_search
        - 其他 → llm_chat
        """
        need_lower = need.lower()
        
        if any(k in need_lower for k in ['代码', '编程', 'python', '写程序', '脚本']):
            return 'code_execute'
        if any(k in need_lower for k in ['画', '图', '生成图像', '画画', '图片']):
            return 'image_generate'
        if any(k in need_lower for k in ['说', '读', '语音', '说话', '念']):
            return 'text_to_speech'
        if any(k in need_lower for k in ['搜索', '查', '找', '上网', '百度']):
            return 'web_search'
        
        return 'llm_chat'
    
    def call(self, tool_name: str, **params) -> Dict:
        """
        调用指定工具。
        
        Args:
            tool_name: 工具名称
            **params: 工具参数
        """
        if tool_name not in self.tools:
            return {
                'success': False,
                'result': None,
                'error': f'未知工具: {tool_name}'
            }
        
        tool = self.tools[tool_name]
        
        # 执行调用
        try:
            result = tool.call(**params)
            
            # 记录调用历史
            call_record = ToolCall(
                tool_name=tool_name,
                params=params,
                result=result.get('result'),
                success=result.get('success', False),
                error=result.get('error')
            )
            self.call_history.append(call_record)
            
            return result
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def auto_call(self, need: str, **params) -> Dict:
        """
        自动路由并调用工具。
        
        根据需求描述自动选择最合适的工具并调用。
        """
        tool_name = self.route_tool(need)
        if not tool_name:
            return {
                'success': False,
                'result': None,
                'error': '无法确定合适的工具'
            }
        
        print(f"[智能路由] 需求'{need}' → 工具'{tool_name}'")
        return self.call(tool_name, **params)
    
    def get_call_history(self, limit: int = 10) -> List[ToolCall]:
        """获取最近N次调用记录"""
        return self.call_history[-limit:]
    
    def export_history(self) -> List[Dict]:
        """导出调用历史（供学习使用）"""
        return [
            {
                'tool_name': c.tool_name,
                'params': c.params,
                'result': str(c.result)[:200] if c.result else None,
                'success': c.success,
                'timestamp': c.timestamp,
                'error': c.error
            }
            for c in self.call_history
        ]


def test_tool_call_engine():
    """测试工具调用引擎"""
    print("=" * 60)
    print("AI工具调用引擎测试")
    print("=" * 60)
    
    # 1. 初始化引擎
    print("\n[1/5] 初始化工具引擎...")
    engine = ToolCallEngine()
    print(f"  已注册 {len(engine.tools)} 个工具")
    
    # 2. 列出工具
    print("\n[2/5] 可用工具列表:")
    for tool_info in engine.list_tools():
        print(f"  - {tool_info['name']}: {tool_info['description']}")
    
    # 3. 智能路由测试
    print("\n[3/5] 智能路由测试:")
    test_needs = [
        "帮我写个Python脚本",
        "画一只小猫",
        "把这段话读出来",
        "搜索一下明天的天气",
        "解释一下量子力学",
    ]
    for need in test_needs:
        tool = engine.route_tool(need)
        print(f"  '{need}' → {tool}")
    
    # 4. 调用测试
    print("\n[4/5] 工具调用测试:")
    
    # 4.1 代码执行
    print("\n  [代码执行]")
    result = engine.call('code_execute', code='print("Hello, 洛神!")\nprint(2 + 3)')
    print(f"    成功: {result['success']}")
    if result['success']:
        print(f"    输出: {result['result']['stdout'].strip()}")
    
    # 4.2 语音合成
    print("\n  [语音合成]")
    result = engine.call('text_to_speech', text='你好，我是洛神！')
    print(f"    成功: {result['success']}")
    if result['success']:
        print(f"    状态: {result['result']['status']}")
    
    # 4.3 图像生成
    print("\n  [图像生成]")
    result = engine.call('image_generate', prompt='一只在草地上奔跑的小狗', style='卡通')
    print(f"    成功: {result['success']}")
    if result['success']:
        print(f"    状态: {result['result']['status']}")
    
    # 4.4 自动调用
    print("\n  [自动调用]")
    result = engine.auto_call('帮我搜索一下人工智能', query='人工智能最新发展')
    print(f"    路由结果: {result.get('result', {}).get('query', 'N/A')}")
    
    # 5. 调用历史
    print("\n[5/5] 调用历史:")
    history = engine.get_call_history()
    print(f"  总调用次数: {len(history)}")
    for i, call in enumerate(history[-3:], 1):
        status = "✓" if call.success else "✗"
        print(f"  {status} [{call.tool_name}] {call.timestamp}")
    
    # 导出
    export = engine.export_history()
    with open('tool_call_history.json', 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    
    print("\n[报告] tool_call_history.json")
    print("=" * 60)


if __name__ == '__main__':
    test_tool_call_engine()
