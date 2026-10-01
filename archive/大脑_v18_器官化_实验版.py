#!/usr/bin/env python3
"""
yinyang_brain_v18 — 按人脑器官结构设计的洛神世界模型
======================================================
将v17o的单一隐状态层拆分为多个功能模块，每个对应一个已知脑区：

  感觉通道 (SensoryChannel)   → V1/A1/S1 初级感觉皮层：特征提取
  联合皮层 (AssociationCortex) → 感觉联合皮层：跨通道模式 + 时序预测（小脑机制保留）
  海马体   (Hippocampus)       → 经验回放缓冲 + 慢速巩固 = 长期记忆
  杏仁核   (Amygdala)          → 情绪标记：高误差事件获得额外巩固权重
  前额叶   (PrefrontalCortex)  → 注意力门控 + 工作记忆选择
  基底节   (BasalGanglia)      → 探索/利用的行为选择（替换IM）
  下丘脑   (Hypothalamus)      → 多维度内稳态驱力（饥饿/困惑/疲劳/好奇...）
  语言接口  (LanguagePort)     → 布洛卡区 + 韦尼克区：先做结构占位，接LLM预留

生理常数：感觉通道数=8（对应v17o），隐状态≤30，蛋白折叠保留
"""

import random, math, hashlib, copy, collections
import numpy as np

# ============ 基本常数 ============
N_SENSE_CHANNELS = 8
MAX_ASSOC_NODES = 30
N_REGIONS = 9
DELAY = 3

# ============ 工具函数 ============
def sigmoid(x):
    try:
        if x > 20: return 1.0
        if x < -20: return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    except:
        return 0.5

def softmax_temperature(logits, temperature=1.0):
    if not logits:
        return []
    mx = max(logits)
    exps = [math.exp((l - mx) / max(temperature, 0.01)) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]

# ============ Pool (环境) ============
class Environment:
    def __init__(self, seed=42):
        self.seed = seed
        self.r = self._mk(seed)
        self._r()
    def _mk(self, s):
        rng = random.Random(s)
        return [{'id': i, 'sigma': round(0.05 + 0.64 * i / 8, 2), 'proprio': round(i / 8, 2),
                 'freq': round(0.03 + 0.12 * rng.random(), 3), 'phase': round(rng.uniform(0, 2 * math.pi), 3),
                 'type': ['sin', 'logistic', 'lorenz', 'vdP', 'saw', 'rossler', 'henon', 'tent', 'cubic'][i],
                 '_s': 0.3, '_t': 0, '_p': 0.0} for i in range(N_REGIONS)]
    def _r(self):
        for r in self.r: r['_s'] = 0.3; r['_t'] = 0; r['_p'] = 0.0
    def reset(self): self._r()
    def step(self):
        for r in self.r: r['_t'] += 1; r['_p'] += r['freq']
    def sense(self, rid):
        r = self.r[rid]; t = r['_t']; ph = r['_p']
        nv = random.Random(hash((self.seed, rid, t)) % (2 ** 31)).gauss(0, 1)
        n = r['sigma'] * nv; s = r['_s']; rt = r['type']
        if rt == 'sin': v = math.sin(ph + r['phase'])
        elif rt == 'logistic': v = 4 * s * (1 - s) if 0 < s < 1 else 0.5
        elif rt == 'lorenz': v = 0.5 + 0.3 * math.tanh(s)
        elif rt == 'vdP': v = s - 0.1 * (s ** 3 - s)
        elif rt == 'saw': v = (ph + r['phase']) % 1.0
        elif rt == 'rossler': v = 0.5 * math.sin(0.7 * ph + s)
        elif rt == 'henon': v = 1 - 1.4 * s ** 2 + 0.3 * s
        elif rt == 'tent': v = 2 * s if s < 0.5 else 2 * (1 - s)
        else: v = s ** 3 - s
        s2 = max(-2, min(2, v + n)); r['_s'] = s2
        return [s2, math.sin(s2 * 2.5) if abs(s2) < 1.5 else 0, math.tanh(s2),
                s2 ** 2 * (1 if s2 > 0 else -1), abs(s2), math.cos(s2 * 1.7),
                (1 if s2 > 0 else -1) * math.sqrt(abs(s2)) if abs(s2) < 2 else 0, r['proprio']]

# ================================================================
#  模块 1: 感觉通道 (初级感觉皮层 V1/A1/S1)
# ================================================================
class SensoryChannel:
    """
    每个感觉通道做独立的初级特征提取。
    生物类比：V1的简单细胞 → 方向/频率选择；A1的声频拓扑映射
    
    当前8个通道只有一种来源（数学信号），所以先不加通道特异性。
    但保留通道索引——当以后接摄像头/麦克风时各自有预滤波器。
    """
    def __init__(self, genome, ch_idx):
        self.ch_idx = ch_idx
        n = genome['n_assoc']
        # 每个通道自己的初级权重 (n_sense → n_assoc 的投影子集)
        # 这里简化为：每个通道有一个"频率偏好"和"增益"
        self.freq_pref = genome['sense_freq'][ch_idx] if ch_idx < len(genome.get('sense_freq', [])) else 1.0
        self.gain = genome['sense_gain'][ch_idx] if ch_idx < len(genome.get('sense_gain', [])) else 1.0
        # 该通道对联合皮层各节点的投射权重
        self.proj_w = genome['sense_proj'][ch_idx][:] if ch_idx < len(genome.get('sense_proj', [])) else [0.0] * n
    
    def process(self, raw_value):
        """初级特征提取：增益 × 非线性"""
        # 类似V1的对比度归一化
        v = raw_value * self.gain
        # 多重尺度特征
        return [math.tanh(v), math.tanh(v * self.freq_pref), math.tanh(v * 0.5)]

# ================================================================
#  模块 2: 联合皮层 + 小脑 (AssociationCortex)
#  保留v17o的蛋白折叠(rw)时序预测机制 = 小脑
#  ================================================================
class AssociationCortex:
    """
    感觉联合皮层 = 跨通道模式识别 + 时序预测（小脑蛋白折叠）
    输入：各通道初级特征的延迟线
    输出：联合隐状态 h_assoc（供前额叶、海马体、读出层使用）
    """
    def __init__(self, genome):
        self.g = genome
        n = genome['n_assoc']
        self.h = [0.0] * n
        self.prev_h = [0.0] * n
        # 延迟线：当前 + DELAY 历史帧
        self.db_history = []
        self.input_vec = []  # 当前展平后的输入（用于skip连接）
        self.err_history = []
    
    def build_input(self, channel_features):
        """
        channel_features: 各通道初级特征拼接后的向量（长度 8×3=24）
        加上延迟历史 → 展平为 input_vec
        """
        flat = []
        for cf in channel_features:
            flat.extend(cf)
        self.db_history.append(flat[:])
        if len(self.db_history) > DELAY:
            self.db_history.pop(0)
        while len(self.db_history) < DELAY:
            self.db_history.insert(0, [0.0] * (N_SENSE_CHANNELS * 3))
        
        result = flat[:]
        for frame in self.db_history:
            result.extend(frame)
        self.input_vec = result[:]
        return result
    
    def fwd(self, channel_features, update=True):
        x = self.build_input(channel_features)
        g = self.g; n = g['n_assoc']
        p = channel_features[0][0] if channel_features else 0  # proprio proxy
        dec = g['dt_decay'][0]; bb = 0
        
        # 感觉输入 → 隐状态（sw: input_dim × n）
        nn = min(len(g['sw_assoc']), len(x))
        nh = [0.0] * n
        for i in range(n):
            a = 0.0
            for j in range(nn):
                a += g['sw_assoc'][j][i] * x[j]
            for k in range(n):
                a += g['rw_assoc'][i][k] * self.h[k]
            nh[i] = math.tanh(a)
        
        if update:
            for i in range(n):
                self.h[i] = (1 - dec) * nh[i] + dec * self.h[i]
        
        # 读出（pw + pb + skip_w 双通路，同v17o）
        pred = [0.0] * N_SENSE_CHANNELS  # 只预测下一帧的原始8通道信号（不是24通道特征）
        for i in range(n):
            for c in range(N_SENSE_CHANNELS):
                pred[c] += self.h[i] * (g['pw_assoc'][i][c] + g['pb_assoc'][c])
        
        n_skip = min(len(g['skip_assoc']), len(x))
        for j in range(n_skip):
            for c in range(N_SENSE_CHANNELS):
                pred[c] += g['skip_assoc'][j][c] * x[j]
        
        if update:
            self.prev_h = self.h[:]
        
        return pred
    
    def protein_fold(self):
        """小脑时序学习（同v17o的蛋白折叠机制，已修正转置）"""
        g = self.g; n = g['n_assoc']
        fold_lr = g.get('fold_lr', g['lr'] * 0.1)
        for k in range(n):
            predicted = 0.0
            for i in range(n):
                predicted += g['rw_assoc'][k][i] * self.prev_h[i]
            if abs(predicted) > 0.01:
                predicted = math.tanh(predicted)
                err = self.h[k] - predicted
                for i in range(n):
                    delta = fold_lr * self.prev_h[i] * err
                    g['rw_assoc'][k][i] += delta
                    g['rw_assoc'][k][i] = max(-1.2, min(1.2, g['rw_assoc'][k][i]))
    
    def reset(self):
        self.h = [0.0] * len(self.h)
        self.prev_h = [0.0] * len(self.prev_h)
        self.db_history = []
        self.input_vec = []
        self.err_history = []

# ================================================================
#  模块 3: 海马体 (Hippocampus) — 经验回放 + 慢速巩固
# ================================================================
class Hippocampus:
    """
    功能1：短期到长期的记忆巩固（experience replay buffer）
    功能2：模式分离（pattern separation）—— 把相似经历编码为可区分的内部表征
    
    缓冲区存 (h_assoc_snapshot, next_h_assoc, action_id, reward_tuple)
    定期从缓冲区随机抽样，驱动慢速权重更新（模拟睡眠/静息态的重播）
    
    生物对应：海马体CA1区的位置细胞 + 齿状回的模式分离
    """
    def __init__(self, genome, buffer_size=128):
        self.g = genome
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)
        self.consolidation_lr = genome.get('lr', 0.08) * 0.05  # 慢速学习（比在线学习慢20倍）
    
    def store(self, h_before, h_after, prediction_error, region_id):
        """存储记忆痕迹 = 状态转移 + 误差"""
        self.buffer.append({
            'h_before': h_before[:],
            'h_after': h_after[:],
            'error': prediction_error,
            'region': region_id,
            'timestamp': len(self.buffer),
            'replay_count': 0,
            'salience': min(1.0, prediction_error * 2.0),  # 高误差=高凸显=易被重放
        })
    
    def replay_batch(self, batch_size=8):
        """
        随机抽样重放 → 驱动慢速权重更新
        生物对应：慢波睡眠中的海马体重播
        """
        if len(self.buffer) < batch_size:
            return 0.0
        
        batch = random.sample(list(self.buffer), batch_size)
        total_loss = 0.0
        
        g = self.g
        n = g['n_assoc']
        lr = self.consolidation_lr
        
        for memory in batch:
            # 用 h_before 预测 h_after（模拟时序巩固）
            h_b = memory['h_before']
            h_a = memory['h_after']
            
            # 计算预测（通过rw_assoc预测）
            pred_h = [0.0] * n
            for k in range(n):
                s = 0.0
                for i in range(n):
                    s += g['rw_assoc'][k][i] * h_b[i]
                pred_h[k] = math.tanh(s) if abs(s) > 0.01 else 0.0
            
            # 误差
            for k in range(n):
                err = h_a[k] - pred_h[k]
                total_loss += err * err
                # 慢速Hebbian更新
                for i in range(n):
                    delta = lr * h_b[i] * err
                    g['rw_assoc'][k][i] += delta
                    g['rw_assoc'][k][i] = max(-1.5, min(1.5, g['rw_assoc'][k][i]))
            
            memory['replay_count'] += 1
        
        return total_loss / max(1, batch_size * n)
    
    def get_stats(self):
        if not self.buffer:
            return {'size': 0, 'avg_salience': 0, 'max_salience': 0}
        saliences = [m['salience'] for m in self.buffer]
        return {
            'size': len(self.buffer),
            'avg_salience': sum(saliences) / len(saliences),
            'max_salience': max(saliences),
            'total_replays': sum(m['replay_count'] for m in self.buffer),
        }
    
    def reset(self):
        self.buffer.clear()

# ================================================================
#  模块 4: 杏仁核 (Amygdala) — 情绪标记 + 记忆增强
# ================================================================
class Amygdala:
    """
    功能：给事件打情绪标签（正/负/惊讶），影响海马体巩固权重
    生物对应：基底外侧杏仁核（BLA）→ 情绪记忆的调制
    
    简单实现：计算每个tick的"惊讶度"（prediction error的异常程度）
    惊讶度高的记忆获得额外的巩固权重
    """
    def __init__(self):
        self.baseline_err = 0.3
        self.err_ema_alpha = 0.05
        self.last_surprise = 0.0
        self.emotional_trace = collections.deque(maxlen=50)
    
    def update(self, prediction_error, region_changed=False):
        """更新基线并返回惊讶度"""
        self.baseline_err = (1 - self.err_ema_alpha) * self.baseline_err + self.err_ema_alpha * prediction_error
        # 惊讶度 = 误差偏离基线的程度（标准化）
        surprise = max(0.0, prediction_error - self.baseline_err) / max(0.01, self.baseline_err)
        # 区域切换也有情绪权重
        if region_changed:
            surprise += 0.3
        surprise = min(5.0, surprise)  # 上限
        self.last_surprise = surprise
        self.emotional_trace.append(surprise)
        return surprise
    
    def get_modulation(self):
        """当前的情绪调制因子（乘到海马体巩固强度上）"""
        return 1.0 + self.last_surprise  # 惊讶→巩固增强
    
    def reset(self):
        self.baseline_err = 0.3
        self.last_surprise = 0.0
        self.emotional_trace.clear()

# ================================================================
#  模块 5: 前额叶皮层 (PrefrontalCortex) — 注意力门控 + 工作记忆
# ================================================================
class PrefrontalCortex:
    """
    功能1：注意力门控（哪些感觉通道进入联合皮层）
    功能2：工作记忆（task set maintenance — 维护当前目标）
    
    生物对应：dlPFC（背外侧）做执行控制，vlPFC（腹外侧）做注意力门控
    
    实现：每个通道生成一个attention gate [0,1]，低于阈值则抑制
    保留Top-K通道的输入，其余置零（类似顶叶的竞争性选择）
    """
    def __init__(self, genome):
        self.g = genome
        self.n_channels = N_SENSE_CHANNELS
        # 注意力权重（每个通道一个）
        self.attn_weights = [1.0] * self.n_channels
        # 工作记忆目标（当前维持的"任务表征"）
        self.task_set = [0.0] * min(8, genome['n_assoc'])
        self.attn_lr = genome.get('lr', 0.08) * 0.3
    
    def compute_attention(self, channel_features, surprise_signal):
        """
        基于通道特征强度和当前杏仁核惊讶度计算注意力门控
        规则：高惊讶 → 注意力分散（所有通道打开）
              低惊讶 → 注意力集中（只开高激活通道）
        """
        # 每个通道的平均激活强度
        channel_energies = []
        for ch_f in channel_features:
            energy = sum(abs(v) for v in ch_f) / max(1, len(ch_f))
            channel_energies.append(energy)
        
        # 惊讶度高 → temperature高 → 门控均匀分散
        temperature = 0.5 + surprise_signal * 0.5
        
        if sum(channel_energies) < 1e-6:
            self.attn_weights = [1.0] * self.n_channels
        else:
            # softmax门控
            self.attn_weights = softmax_temperature(
                [e * temperature for e in channel_energies],
                temperature=max(0.1, temperature)
            )
            # 重新归一化到[0,1]范围（softmax在temperature>1时更均匀）
            # 改为：能量低的通道被抑制
            mx = max(self.attn_weights)
            if mx > 0:
                self.attn_weights = [w / mx for w in self.attn_weights]
        
        return self.attn_weights
    
    def apply_gate(self, channel_features):
        """将注意力门控施加到通道特征上"""
        gated = []
        for i, ch_f in enumerate(channel_features):
            w = self.attn_weights[i] if i < len(self.attn_weights) else 1.0
            gated.append([v * w for v in ch_f])
        return gated
    
    def reset(self):
        self.attn_weights = [1.0] * self.n_channels
        self.task_set = [0.0] * len(self.task_set)

# ================================================================
#  模块 6: 基底神经节 (BasalGanglia) — 探索/利用行为选择
# ================================================================
class BasalGanglia:
    """
    功能：决定下一步的行动策略——探索新区域 vs 利用已知区域
    
    生物对应：纹状体多巴胺调节的Go/No-Go通路
    - 直接通路（"Go"）= 前额叶当前目标 → 维持当前区域
    - 间接通路（"No-Go"）= 新奇信号 → 切换到新区域
    
    状态：维护每个区域的"熟悉度"(familiarity)
    熟悉度高 → exploitation信号强；新奇高 → exploration信号强
    """
    def __init__(self):
        self.familiarity = [0.0] * N_REGIONS
        self.exploration_drive = 0.3
        self.last_switch_tick = 0
        self.current_tick = 0
        self.switch_history = collections.deque(maxlen=20)
        self.boredom = 0.0
    
    def update(self, region_id, prediction_error, tick_gap):
        self.current_tick += tick_gap
        
        # 熟悉度增长（每次访问累加，缓慢衰减）
        for i in range(N_REGIONS):
            self.familiarity[i] *= 0.998  # 缓慢遗忘
        self.familiarity[region_id] += 0.01
        
        # 误差高 → 新奇感 → 探索驱动增加
        novelty_signal = min(1.0, max(0, prediction_error - 0.3))
        self.exploration_drive += novelty_signal * 0.005
        
        # 无聊度：在原区停留太久且误差低 → 无聊积累
        time_in_region = tick_gap
        if prediction_error < 0.2:
            self.boredom += 0.001 * time_in_region
        else:
            self.boredom *= 0.95  # 有挑战 → 无聊消退
        
        self.boredom = min(1.0, self.boredom)
        self.exploration_drive = min(1.0, self.exploration_drive)
    
    def decide_region(self, current_region, forced_explore=False):
        """返回下一个区域ID"""
        if forced_explore:
            # 强制探索 → 选熟悉度最低的区域
            return min(range(N_REGIONS), key=lambda r: self.familiarity[r])
        
        # 探索概率
        p_explore = self.exploration_drive + self.boredom * 0.3
        p_explore = min(0.3, max(0.01, p_explore))  # 1%-30%
        
        if random.random() < p_explore:
            # 探索：选熟悉度最低的几个之一（带softmax噪声）
            candidates = list(range(N_REGIONS))
            weights = [1.0 / (0.01 + f) for f in self.familiarity]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            return np.random.choice(candidates, p=probs)
        else:
            return current_region
    
    def reset(self):
        self.familiarity = [0.0] * N_REGIONS
        self.exploration_drive = 0.3
        self.boredom = 0.0

# ================================================================
#  模块 7: 下丘脑 (Hypothalamus) — 多维度内稳态驱力
# ================================================================
class Hypothalamus:
    """
    功能：维护多个并行驱力变量，每个变量有设定点(setpoint)和当前值
    偏差 → 产生"不适感" → 影响行为
    
    维度：
    1. curiosity（好奇心驱力）— 预测误差长期低时积累
    2. coherence（一致性驱力）— 模型预测长期不准时积累
    3. engagement（参与驱力）— 刺激变化率长期低时积累
    4. mastery（掌控驱力）— 已探索区域比例低时积累
    
    生物对应：下丘脑的饥饿/渴/体温调节 → 并行内稳态
    """
    def __init__(self):
        self.drives = {
            'curiosity': {'setpoint': 0.0, 'value': 0.5, 'min': 0, 'max': 1.0, 'decay': 0.995, 'fill_rate': 0.003},
            'coherence': {'setpoint': 0.3, 'value': 0.0, 'min': 0, 'max': 1.0, 'decay': 0.99, 'fill_rate': 0.005},
            'engagement': {'setpoint': 0.0, 'value': 0.5, 'min': 0, 'max': 1.0, 'decay': 0.99, 'fill_rate': 0.002},
            'mastery': {'setpoint': 0.0, 'value': 0.9, 'min': 0, 'max': 1.0, 'decay': 0.998, 'fill_rate': 0.001},
        }
        self.history = collections.deque(maxlen=200)
    
    def update(self, prediction_error, region_change, region_coverage, dt=1):
        """每个tick更新所有驱力变量"""
        d = self.drives
        
        # curiosity: 误差低时上升（无聊→好奇），误差高时下降（刺激满足）
        if prediction_error < 0.2:
            d['curiosity']['value'] += d['curiosity']['fill_rate'] * dt
        else:
            d['curiosity']['value'] -= prediction_error * 0.1 * dt
        
        # coherence: 误差高时上升，误差低时下降（已经掌控→无需增强）
        if prediction_error > 0.5:
            d['coherence']['value'] += d['coherence']['fill_rate'] * dt
        else:
            d['coherence']['value'] -= 0.002 * dt
        
        # engagement: 区域切换或刺激变化大时满足，否则积累
        if region_change:
            d['engagement']['value'] -= 0.05 * dt
        else:
            d['engagement']['value'] += d['engagement']['fill_rate'] * dt
        
        # mastery: 高覆盖率时满足（已经知道这个区域了）
        mastery_gap = 1.0 - region_coverage  # 覆盖率低→掌控缺失
        d['mastery']['value'] = d['mastery']['decay'] * d['mastery']['value'] + (1 - d['mastery']['decay']) * mastery_gap
        
        # 所有变量衰减到min
        for k in d:
            d[k]['value'] = max(d[k]['min'], min(d[k]['max'], d[k]['value']))
            d[k]['value'] *= d[k]['decay']
        
        state = {k: v['value'] for k, v in d.items()}
        self.history.append(state)
        return state
    
    def get_urgency(self):
        """总体不适感 = 各驱力加权和"""
        weights = {'curiosity': 1.0, 'coherence': 1.5, 'engagement': 0.8, 'mastery': 0.6}
        urgency = sum(self.drives[k]['value'] * w for k, w in weights.items())
        return urgency
    
    def get_dominant_drive(self):
        """当前最强的驱力名称"""
        return max(self.drives, key=lambda k: self.drives[k]['value'])
    
    def reset(self):
        for k in self.drives:
            self.drives[k]['value'] = 0.3
        self.history.clear()

# ================================================================
#  模块 8: 语言接口 (LanguagePort) — 布洛卡区+韦尼克区（结构占位）
# ================================================================
class LanguagePort:
    """
    先做结构：能接收LLM输入、能将内部状态格式化为prompt。
    实际语言能力需要接LLM API后才能激活。
    """
    def __init__(self):
        self.llm_enabled = False
        self.prompt_template = (
            "以下是一个数字生命体的内部状态：\n"
            "{internal_state}\n"
            "请用第一人称描述这个生命体当前的感受和需求："
        )
        self.last_output = ""
    
    def format_state(self, h_assoc, surprise, drives, prediction_error):
        """格式化内部状态为文本（供LLM使用）"""
        state_str = (
            f"internal_activation: {[round(v, 3) for v in h_assoc[:8]]}...\n"
            f"surprise: {surprise:.3f}\n"
            f"prediction_error: {prediction_error:.4f}\n"
            f"drives: { {k: round(v, 3) for k, v in drives.items()} }\n"
        )
        return self.prompt_template.format(internal_state=state_str)
    
    def generate_speech(self, h_assoc, surprise, drives, prediction_error):
        """如果LLM未启用，返回占位文本"""
        if not self.llm_enabled:
            return f"[语言接口未激活] 惊讶度={surprise:.2f}, 驱力={self.get_dominant_drive(drives)}"
        # 实际使用时调用LLM
        return "LLM输出占位"
    
    def get_dominant_drive(self, drives):
        if not drives:
            return "未知"
        return max(drives, key=drives.get)
    
    def reset(self):
        self.last_output = ""

# ================================================================
#  基因组 (Genome) — 所有模块的权重和参数
# ================================================================
class Genome:
    """有机体的完整基因组，包含所有脑区模块的参数"""
    
    @staticmethod
    def random():
        n_assoc = random.randint(12, MAX_ASSOC_NODES)
        input_dim = N_SENSE_CHANNELS * 3 * (1 + DELAY)  # 感觉通道特征(24) × 延迟帧(4)
        
        g = {
            # 联合皮层（小脑蛋白折叠）
            'n_assoc': n_assoc,
            'sw_assoc': [[random.uniform(-0.3, 0.3) for _ in range(n_assoc)] for __ in range(input_dim)],
            'pw_assoc': [[random.uniform(-0.3, 0.3) for _ in range(N_SENSE_CHANNELS)] for __ in range(n_assoc)],
            'rw_assoc': [[random.uniform(-0.04, 0.04) for _ in range(n_assoc)] for __ in range(n_assoc)],
            'skip_assoc': [[random.uniform(-0.15, 0.15) for _ in range(N_SENSE_CHANNELS)] for __ in range(input_dim)],
            'pb_assoc': [random.uniform(-0.2, 0.2) for _ in range(N_SENSE_CHANNELS)],
            'dt_decay': [random.uniform(0.3, 0.9)],
            
            # 学习参数
            'lr': random.uniform(0.03, 0.12),
            'fold_lr': random.uniform(0.015, 0.06),
            
            # 感觉通道参数
            'sense_freq': [random.uniform(0.5, 2.0) for _ in range(N_SENSE_CHANNELS)],
            'sense_gain': [random.uniform(0.7, 1.5) for _ in range(N_SENSE_CHANNELS)],
            'sense_proj': [[random.uniform(-0.1, 0.1) for _ in range(N_SENSE_CHANNELS)] for __ in range(n_assoc)],
            
            # 基因组自身信息
            'lid': hashlib.sha256(str(random.random()).encode()).hexdigest()[:8],
        }
        
        # 蛋白折叠自连接（类似v17o的对角线初始化）
        for i in range(n_assoc):
            g['rw_assoc'][i][i] = random.uniform(0.3, 1.0)
        
        return g
    
    @staticmethod
    def crossover(a, b):
        """模块级交叉"""
        # 简化：选较大的n_assoc
        n = max(a['n_assoc'], b['n_assoc'])
        input_dim = N_SENSE_CHANNELS * 3 * (1 + DELAY)
        
        ch = {
            'n_assoc': n,
            'sw_assoc': [], 'pw_assoc': [], 'rw_assoc': [], 'skip_assoc': [],
            'pb_assoc': [0.0] * N_SENSE_CHANNELS,
            'dt_decay': a['dt_decay'][:],
            'lr': a['lr'] if random.random() < 0.5 else b['lr'],
            'fold_lr': a['fold_lr'] if random.random() < 0.5 else b['fold_lr'],
            'sense_freq': [a['sense_freq'][i] if random.random() < 0.5 else b['sense_freq'][i] for i in range(N_SENSE_CHANNELS)],
            'sense_gain': [a['sense_gain'][i] if random.random() < 0.5 else b['sense_gain'][i] for i in range(N_SENSE_CHANNELS)],
            'sense_proj': [[] for _ in range(n)],
            'lid': hashlib.sha256(str(random.random()).encode()).hexdigest()[:8],
        }
        
        # 基因级交叉：sw/pw/rw/skip
        for j in range(input_dim):
            src = a if random.random() < 0.5 else b
            row = [0.0] * n
            for i in range(n):
                si = min(i, src['n_assoc'] - 1)
                if j < len(src['sw_assoc']) and si < len(src['sw_assoc'][j]):
                    row[i] = src['sw_assoc'][j][si]
            ch['sw_assoc'].append(row)
        
        for i in range(n):
            src = a if random.random() < 0.5 else b
            si = min(i, src['n_assoc'] - 1)
            if si < len(src['pw_assoc']):
                ch['pw_assoc'].append(src['pw_assoc'][si][:])
            else:
                ch['pw_assoc'].append([0.0] * N_SENSE_CHANNELS)
        
        for i in range(n):
            row = [0.0] * n
            src = a if random.random() < 0.5 else b
            si = min(i, src['n_assoc'] - 1)
            if si < len(src['rw_assoc']):
                for k in range(n):
                    sk = min(k, src['n_assoc'] - 1)
                    if sk < len(src['rw_assoc'][si]):
                        row[k] = src['rw_assoc'][si][sk]
            ch['rw_assoc'].append(row)
        
        for j in range(input_dim):
            src = a if random.random() < 0.5 else b
            if j < len(src['skip_assoc']):
                ch['skip_assoc'].append(src['skip_assoc'][j][:])
            else:
                ch['skip_assoc'].append([0.0] * N_SENSE_CHANNELS)
        
        # pb: 整段继承
        src = a if random.random() < 0.5 else b
        ch['pb_assoc'] = src['pb_assoc'][:N_SENSE_CHANNELS]
        
        return ch
    
    @staticmethod
    def mutate(g, rate=0.05):
        """各模块独立突变"""
        # sw
        for j in range(len(g['sw_assoc'])):
            for i in range(len(g['sw_assoc'][j])):
                if random.random() < rate:
                    g['sw_assoc'][j][i] += random.gauss(0, 0.08)
                    g['sw_assoc'][j][i] = max(-3, min(3, g['sw_assoc'][j][i]))
        # pw
        for i in range(len(g['pw_assoc'])):
            for c in range(N_SENSE_CHANNELS):
                if random.random() < rate:
                    g['pw_assoc'][i][c] += random.gauss(0, 0.08)
                    g['pw_assoc'][i][c] = max(-3, min(3, g['pw_assoc'][i][c]))
        # rw（小脑蛋白折叠）
        for i in range(len(g['rw_assoc'])):
            for k in range(len(g['rw_assoc'][i])):
                if random.random() < rate * 0.7:
                    g['rw_assoc'][i][k] += random.gauss(0, 0.04)
                    g['rw_assoc'][i][k] = max(-1.5, min(1.5, g['rw_assoc'][i][k]))
        # skip
        for j in range(len(g['skip_assoc'])):
            for c in range(N_SENSE_CHANNELS):
                if random.random() < rate:
                    g['skip_assoc'][j][c] += random.gauss(0, 0.08)
                    g['skip_assoc'][j][c] = max(-2, min(2, g['skip_assoc'][j][c]))
        # sense params
        if random.random() < 0.3:
            idx = random.randint(0, N_SENSE_CHANNELS - 1)
            g['sense_freq'][idx] += random.gauss(0, 0.1)
            g['sense_freq'][idx] = max(0.1, min(5.0, g['sense_freq'][idx]))
        if random.random() < 0.3:
            idx = random.randint(0, N_SENSE_CHANNELS - 1)
            g['sense_gain'][idx] += random.gauss(0, 0.1)
            g['sense_gain'][idx] = max(0.1, min(3.0, g['sense_gain'][idx]))
        if random.random() < 0.2:
            g['lr'] = max(0.02, min(0.2, g['lr'] + random.gauss(0, 0.02)))
            g['fold_lr'] = max(0.005, min(0.1, g['fold_lr'] + random.gauss(0, 0.015)))
        return g

# ================================================================
#  有机体 (Organism) — 组装所有模块
# ================================================================
class Organism:
    def __init__(self, genome=None):
        self.g = genome or Genome.random()
        
        # 组装脑区模块
        self.cortex = AssociationCortex(self.g)          # 联合皮层（小脑蛋白折叠）
        self.hippo = Hippocampus(self.g, buffer_size=128) # 海马体
        self.amygdala = Amygdala()                         # 杏仁核
        self.pfc = PrefrontalCortex(self.g)                 # 前额叶
        self.basal = BasalGanglia()                        # 基底节
        self.hypo = Hypothalamus()                         # 下丘脑
        self.language = LanguagePort()                     # 语言接口
        
        # 感觉通道
        self.channels = [SensoryChannel(self.g, i) for i in range(N_SENSE_CHANNELS)]
        
        self.age = 0
        self.alive = True
        self._ib = 1.0  # verifier折扣
        self._last_region = 0
        self._tick_count = 0
    
    def tick(self, env):
        """
        完整的一个时间步：感觉→联合→预测→学习→记忆→情绪→注意→行为
        返回: (prediction_error, region_changed)
        """
        # 1. 环境 → 当前区域
        current_region = self.basal.decide_region(self._last_region)
        region_changed = (current_region != self._last_region)
        self._last_region = current_region
        
        # 2. 环境 → 原始感官信号
        raw_sense = env.sense(current_region)
        env.step()
        
        # 3. 感觉通道初级特征提取
        channel_features = []
        for i, ch in enumerate(self.channels):
            if i < len(raw_sense):
                features = ch.process(raw_sense[i])
            else:
                features = [0.0, 0.0, 0.0]
            channel_features.append(features)
        
        # 4. 前额叶注意力门控
        surprise = self.amygdala.last_surprise
        self.pfc.compute_attention(channel_features, surprise)
        gated_features = self.pfc.apply_gate(channel_features)
        
        # 5. 联合皮层前向+预测
        h_before = self.cortex.h[:]
        pred = self.cortex.fwd(gated_features, update=True)
        
        # 6. 下一帧实际值 → 计算误差
        next_sense = env.sense(current_region)
        env.step()
        mse = sum((next_sense[j] - pred[j]) ** 2 for j in range(N_SENSE_CHANNELS)) / N_SENSE_CHANNELS
        self.cortex.err_history.append(mse)
        
        # 7. 联合皮层学习（pw + skip + 蛋白折叠）
        self.cortex.learn(pred, next_sense) if hasattr(self.cortex, 'learn') else self._legacy_learn(pred, next_sense)
        
        # 8. 杏仁核 → 情绪标记
        surprise = self.amygdala.update(mse, region_changed)
        
        # 9. 海马体 → 存储记忆
        h_after = self.cortex.h[:]
        self.hippo.store(h_before, h_after, mse, current_region)
        
        # 10. 下丘脑 → 多维度驱力更新
        region_coverage = sum(1 for f in self.basal.familiarity if f > 0.05) / N_REGIONS
        drives = self.hypo.update(mse, region_changed, region_coverage, dt=1)
        
        # 11. 基底节 → 更新探索/利用状态
        self.basal.update(current_region, mse, self._tick_count)
        
        # 12. 语言接口（占用）
        if self._tick_count % 50 == 0:
            speech = self.language.generate_speech(
                self.cortex.h, surprise, drives, mse
            )
            self._last_speech = speech
        
        self._tick_count += 1
        self.age += 1
        
        return mse, region_changed
    
    def _legacy_learn(self, pred, actual):
        """联合皮层学习（pw + skip + 蛋白折叠）"""
        g = self.g; n = g['n_assoc']; lr = g['lr']
        errs = [actual[j] - pred[j] for j in range(N_SENSE_CHANNELS)]
        
        mse = sum(e * e for e in errs) / N_SENSE_CHANNELS
        
        # pw/pb 学习
        for i in range(n):
            for c in range(N_SENSE_CHANNELS):
                g['pw_assoc'][i][c] += lr * self.cortex.h[i] * errs[c]
                g['pw_assoc'][i][c] = max(-3, min(3, g['pw_assoc'][i][c]))
        for c in range(N_SENSE_CHANNELS):
            g['pb_assoc'][c] += lr * 1.5 * errs[c]
            g['pb_assoc'][c] = max(-5, min(5, g['pb_assoc'][c]))
        
        # skip 学习
        x = self.cortex.input_vec
        n_skip = min(len(g['skip_assoc']), len(x))
        for j in range(n_skip):
            for c in range(N_SENSE_CHANNELS):
                g['skip_assoc'][j][c] += lr * 0.5 * x[j] * errs[c]
                g['skip_assoc'][j][c] = max(-2, min(2, g['skip_assoc'][j][c]))
        
        # 蛋白折叠（小脑时序学习）
        self.cortex.protein_fold()
        return abs(mse)
    
    def sleep(self, n_replays=20):
        """离线巩固：海马体重播驱动慢速权重更新"""
        total_loss = 0.0
        for _ in range(n_replays):
            loss = self.hippo.replay_batch(batch_size=8)
            total_loss += loss
        return total_loss / max(1, n_replays)
    
    def reset(self):
        self.cortex.reset()
        self.hippo.reset()
        self.amygdala.reset()
        self.pfc.reset()
        self.basal.reset()
        self.hypo.reset()
        self.language.reset()
        self._tick_count = 0
        self.age = 0
    
    def eval_fitness(self, env_seed=42, ticks=500):
        """评估适应度"""
        env = Environment(seed=env_seed)
        env.reset()
        self.reset()
        
        # 预热
        for _ in range(30):
            raw = env.sense(random.randint(0, 8))
            env.step()
        
        total_mse = 0.0
        for t in range(ticks):
            mse, _ = self.tick(env)
            total_mse += mse
        
        avg_mse = total_mse / ticks
        return 1.0 / (1.0 + avg_mse) * self._ib

# ================================================================
#  演化引擎 (简化版)
# ================================================================
class EngineV18:
    def __init__(self, n_org=20, n_gen=200, seed=42):
        self.n_org = n_org
        self.n_gen = n_gen
        self.pop = [Organism() for _ in range(n_org)]
        self.hof = []
        self.best_ever = 0.0
    
    def evolve(self):
        for gen in range(self.n_gen):
            # 评估
            fitnesses = []
            for org in self.pop:
                f = org.eval_fitness(env_seed=gen * 7 + 1, ticks=300)
                fitnesses.append(f)
                if f > self.best_ever:
                    self.best_ever = f
            
            bf = max(fitnesses) if fitnesses else 0
            af = sum(fitnesses) / len(fitnesses) if fitnesses else 0
            
            if gen % 20 == 0 or gen == self.n_gen - 1:
                print(f"Gen {gen:3d}  fit={bf:.4f}/{af:.4f}  best={self.best_ever:.4f}")
            
            if gen == self.n_gen - 1:
                break
            
            # 精英保留
            idx_sorted = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            elites = [self.pop[idx_sorted[i]] for i in range(min(3, len(self.pop)))]
            
            # 生成下一代
            next_gen = []
            for e in elites:
                next_gen.append(Organism(copy.deepcopy(e.g)))
            
            while len(next_gen) < self.n_org:
                # 选择父母
                i1 = random.choices(range(len(self.pop)), weights=[max(0.01, f) for f in fitnesses], k=1)[0]
                i2 = random.choices(range(len(self.pop)), weights=[max(0.01, f) for f in fitnesses], k=1)[0]
                
                # 交叉+突变
                child_g = Genome.crossover(self.pop[i1].g, self.pop[i2].g)
                Genome.mutate(child_g, rate=0.05)
                next_gen.append(Organism(child_g))
            
            self.pop = next_gen
        
        return self.pop

# ================================================================
#  主程序
# ================================================================
if __name__ == '__main__':
    print("yinyang_brain_v18 — 人脑器官结构洛神")
    print(f"模块数: 8个（感觉通道+联合皮层+海马体+杏仁核+前额叶+基底节+下丘脑+语言接口）")
    print(f"额外参数: attention gate, 多维度驱力, 经验回放, 情绪调制")
    print()
    
    # 测试单有机体
    print("--- --- 单有机体测试 --- ---")
    org = Organism()
    env = Environment(seed=42)
    env.reset()
    
    for i in range(300):
        mse, rc = org.tick(env)
        if i % 50 == 0:
            print(f"tick {i:3d}  mse={mse:.4f}  surprise={org.amygdala.last_surprise:.3f}  "
                  f"drive={org.hypo.get_dominant_drive()}  attn=[{org.pfc.attn_weights[0]:.2f},{org.pfc.attn_weights[1]:.2f},...]")
    
    # 睡眠巩固
    print("\n--- --- 睡眠巩固 --- ---")
    initial_hippo = org.hippo.get_stats()
    print(f"海马体状态: {initial_hippo}")
    sleep_loss = org.sleep(n_replays=30)
    print(f"睡眠后平均重播损失: {sleep_loss:.6f}")
    final_hippo = org.hippo.get_stats()
    print(f"巩固后重放次数: {final_hippo['total_replays']}")
    
    # fitness评估
    print("\n--- --- Fitness评估 --- ---")
    f = org.eval_fitness(ticks=300)
    print(f"适应度: {f:.4f}")
    
    # 演化测试
    print("\n--- --- 演化测试 --- ---")
    eng = EngineV18(n_org=15, n_gen=100, seed=42)
    eng.evolve()
    print(f"\n最终最佳: {eng.best_ever:.4f}")
