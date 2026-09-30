"""感知处理器 · AI孩子的摄像头输入

输入: 视频文件或摄像头帧流
输出: 四脑物理驱动物（非语义分类）
  · 光 = 环境亮度变化
  · 力 = 帧间运动能量
  · 声 = 低频振动代理（色彩脉动频率）
  · 温 = 肤色像素占比（"有人在"的替代信号）

不使用LLM、不使用语义理解、不输出物体名称。
"""
import numpy as np
import imageio
from pathlib import Path


class PerceptionProcessor:
    """从视频文件或摄像头 → 引擎环境驱动物"""
    
    def __init__(self, source=None, skin_hsv_range=None):
        视频源 = source or "/mnt/data/catpaw/home/workspace/洛神赋1.1/video_input/teacher_speak.mp4"
        视频路径 = Path(视频源)
        if not 视频路径.exists():
            raise FileNotFoundError(f"视频源不存在: {视频源}")
        self.源 = 视频源
        self.帧读器 = imageio.get_reader(视频源, "ffmpeg")
        self.帧率 = self.帧读器.get_meta_data().get("fps", 30)
        self.帧索引 = 0
        self.上一帧灰 = None
        
        # 肤色HSV范围（H:[0-180], S:[0-1], V:[0-1]）
        if skin_hsv_range is None:
            self.skin_hsv_range = (
                np.array([0, 0.06, 0.20]),     # H低, S低容忍阴影, V=0.20
                np.array([35, 0.70, 1.0])      # H扩展, S宽, V=1.0
            )
        else:
            self.skin_hsv_range = (
                np.array(skin_hsv_range[0]),
                np.array(skin_hsv_range[1])
            )
    
    def 读帧(self):
        """读一帧，读取失败返回None"""
        try:
            帧 = self.帧读器.get_data(self.帧索引)
            self.帧索引 += 1
            return 帧
        except IndexError:
            return None
    
    def _帧转灰度(self, 帧):
        """0-255 RGB → 0-1 float 灰度"""
        return np.mean(帧.astype(np.float32), axis=2) / 255.0
    
    def _帧转HSV(self, 帧):
        """简单HSV转换（无OpenCV时估算）。用RGB最大/最小构造。"""
        rgb = 帧.astype(np.float32) / 255.0
        V = np.max(rgb, axis=2)
        min_rgb = np.min(rgb, axis=2)
        差 = V - min_rgb
        # 避免除零
        差_safe = np.where(差 < 1e-8, 1e-8, 差)
        S = np.where(V < 1e-8, 0.0, 差_safe / V)
        # 简化H：用RGB最大通道算
        H = np.zeros_like(V)
        mask_r = rgb[:,:,0] == V
        G_rel = rgb[:,:,1]
        B_rel = rgb[:,:,2]
        H = np.where(mask_r, (G_rel - B_rel) / np.maximum(差_safe, 1e-8),
            np.where(rgb[:,:,1] == V, 2.0 + (B_rel - rgb[:,:,0]) / np.maximum(差_safe, 1e-8),
                4.0 + (rgb[:,:,0] - G_rel) / np.maximum(差_safe, 1e-8))) * 60.0
        H = np.where(差 < 1e-8, 0.0, H) % 180.0
        return H, S, V
    
    def _皮肤占比(self, 帧):
        """肤色像素占总像素比例（0.0-1.0）"""
        H, S, V = self._帧转HSV(帧)
        lo, hi = self.skin_hsv_range
        皮肤掩码 = (
            (H >= lo[0]) & (H <= hi[0]) &
            (S >= lo[1]) & (S <= hi[1]) &
            (V >= lo[2]) & (V <= hi[2])
        )
        return np.mean(皮肤掩码)
    
    def 处理(self):
        """读一帧，返回驱动物字典；帧耗尽返回None"""
        帧 = self.读帧()
        if 帧 is None:
            return None
        
        灰 = self._帧转灰度(帧)
        
        # 光 = 帧平均亮度 (0-1)
        光 = float(np.mean(灰))
        
        # 力 = 帧间运动能量（与上一帧灰度差的绝对值均值）
        if self.上一帧灰 is not None:
            motion = float(np.mean(np.abs(灰 - self.上一帧灰)))
        else:
            motion = 0.0
        力 = motion
        
        # 声 = 低频色彩脉冲：以"红通道标准差/绿通道标准差"变化频率做代理
        红 = 帧[:,:,0].astype(np.float32) / 255.0
        绿 = 帧[:,:,1].astype(np.float32) / 255.0
        蓝 = 帧[:,:,2].astype(np.float32) / 255.0
        脉动_proxy = float(np.std(红) / (np.std(绿) + 1e-8) + np.std(蓝) - np.std(绿))
        声 = abs(脉动_proxy) * 100  # 缩放到与引擎环境量级匹配
        
        # 温 = 肤色占比 → "有人在"的替代信号 → 0-1 映射到 280-310K
        皮肤 = self._皮肤占比(帧)
        温k = 280.0 + 30.0 * 皮肤  # 无人=280K（冷），有人=310K（暖）
        
        结果 = {
            "光": 光,
            "力": 力,
            "声": min(声, 100.0),  # 限幅
            "温": 温k,
            "皮肤": 皮肤,
            "帧索引": self.帧索引,
        }
        
        self.上一帧灰 = 灰
        return 结果
    
    def close(self):
        self.帧读器.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


if __name__ == "__main__":
    """快速自检：跑前100帧，输出驱动物"""
    import numpy as np
    p = PerceptionProcessor()
    for _ in range(100):
        r = p.处理()
        if r is None:
            print("帧耗尽")
            break
        print(f"帧={r['帧索引']:3d} 光={r['光']:.4f} 力={r['力']:.5f} 声={r['声']:.2f} 温={r['温']:.1f}K 皮肤={r['皮肤']:.4f}")
    p.close()
    print("=== 感知处理器自检完成 ===")
