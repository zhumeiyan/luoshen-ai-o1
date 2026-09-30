"""表达选择器 · AI孩子的屏幕输出

输入: 引擎内部状态（饿、醒、筋数、最大位、哭合计…）
输出: 屏幕绘图形素（色块+几何图案）
目的: 让外部应答者能读出婴儿在"要什么"

不使用LLM、不输出文本。纯状态→图形映射。
"""
import numpy as np
from PIL import Image, ImageDraw


class ExpressionSelector:
    """引擎状态 → 屏幕显示"""
    
    # 调色板
    COLOR_QUIET = (80, 180, 120)    # 安静 = 绿
    COLOR_WANT  = (240, 180, 60)    # 想要 = 橙黄
    COLOR_CRY   = (220, 60, 60)     # 哭 = 红
    COLOR_SLEEP = (60, 80, 160)     # 睡/关闭 = 蓝灰
    COLOR_DEAD  = (40, 40, 40)      # 极端 = 近黑
    
    def __init__(self, width=200, height=200):
        self.w = width
        self.h = height
    
    def _饿_to_color(self, 饿, 醒, 哭速):
        """根据饿度、醒状态、哭速决定主色"""
        if not 醒:
            return self.COLOR_SLEEP
        if 饿 < 6.0:
            return self.COLOR_QUIET
        if 饿 < 12.0:
            # 6-12 之间：从绿渐变到橙黄
            t = (饿 - 6.0) / 6.0
            return tuple(int(a + (b-a)*t) for a,b in zip(self.COLOR_QUIET, self.COLOR_WANT))
        # >=12：橙→红
        t = min(1.0, (饿 - 12.0) / 12.0)
        return tuple(int(a + (b-a)*t) for a,b in zip(self.COLOR_WANT, self.COLOR_CRY))
    
    def _位_to_size(self, 最大位, 饿):
        """决定中心圆形大小：激活越高/饿越高→越大越急"""
        base = 20
        if 最大位 > 0:
            base += min(60, 最大位 * 2)
        if 饿 > 10:
            base += int((饿 - 10) * 3)
        return min(base, 90)
    
    def _哭_to_jagged(self, 哭合计, 帧数):
        """哭合计决定边缘抖动：哭越多越尖锐"""
        if 哭合计 == 0:
            return 0
        # 每100声哭 → +1 抖动等级，上限5
        return min(5, 哭合计 // 100)
    
    def render(self, 引擎状态: dict):
        """
        引擎状态包含: 饿, 醒, 筋数, 最大位, 哭合计, 时刻
        返回: PIL Image（可保存为PNG或直接显示）
        """
        饿 = 引擎状态.get('饿', 0.0)
        醒 = 引擎状态.get('醒', False)
        筋数 = 引擎状态.get('筋数', 0)
        最大位 = 引擎状态.get('最大位', 0)
        哭合计 = 引擎状态.get('哭合计', 0)
        时刻 = 引擎状态.get('时刻', 0)
        
        主色 = self._饿_to_color(饿, 醒, 哭合计)
        圆径 = self._位_to_size(最大位, 饿)
        抖动 = self._哭_to_jagged(哭合计, 时刻)
        
        img = Image.new('RGB', (self.w, self.h), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        
        cx, cy = self.w // 2, self.h // 2
        
        if 抖动 == 0:
            # 圆：安静/满足
            draw.ellipse([cx-圆径, cy-圆径, cx+圆径, cy+圆径], fill=主色)
        else:
            # 锯齿：哭/痛苦
            points = []
            n_spikes = 12 + 抖动 * 4
            for i in range(n_spikes * 2):
                angle = 2 * np.pi * i / (n_spikes * 2)
                if i % 2 == 0:
                    r = 圆径 + 抖动 * 4
                else:
                    r = max(5, 圆径 - 抖动 * 3)
                x = cx + int(r * np.cos(angle))
                y = cy + int(r * np.sin(angle))
                points.append((x, y))
            draw.polygon(points, fill=主色)
        
        # 底条：筋数指示（网络密度可视化）
        筋亮条 = min(self.w, int(筋数 / 30))  # 约30筋/pixel宽
        draw.rectangle([0, self.h-8, 筋亮条, self.h], fill=(200, 200, 200))
        
        return img
    
    def render_to_file(self, 引擎状态: dict, 路径):
        img = self.render(引擎状态)
        img.save(路径)
        return 路径


if __name__ == "__main__":
    """自检：模拟三种状态画三张图"""
    sel = ExpressionSelector()
    
    安静 = {'饿': 3.0, '醒': True, '筋数': 2400, '最大位': 5, '哭合计': 0, '时刻': 5000}
    饿揪 = {'饿': 9.0, '醒': True, '筋数': 1800, '最大位': 12, '哭合计': 0, '时刻': 15000}
    痛哭 = {'饿': 18.0, '醒': True, '筋数': 8400, '最大位': 39, '哭合计': 4000, '时刻': 70000}
    
    sel.render_to_file(安静, '/tmp/abai_test_quiet.png')
    sel.render_to_file(饿揪, '/tmp/abai_test_want.png')
    sel.render_to_file(痛哭, '/tmp/abai_test_cry.png')
    print("三张测试图已写入 /tmp/abai_test_*.png")
    print("安静=绿圆, 饿揪=橙圆, 痛哭=红锯齿")
