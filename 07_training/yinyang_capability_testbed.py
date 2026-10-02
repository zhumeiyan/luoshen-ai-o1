#!/usr/bin/env python3
"""
yinyang_capability_testbed — 多能力世界模型测试床
====================================================
同时测试5种高级认知能力:
  1. 物体恒存性 (Object Permanence): 物体短暂消失后仍记得位置
  2. 物理推理 (Physical Reasoning): 碰撞/重力/弹跳的动力学预测
  3. 空间关系 (Spatial Relations): 物体间的相对位置/方向
  4. 生成预测 (Generative Prediction): 多步预测未来轨迹
  5. 反事实推演 (Counterfactual): "如果...会怎样"的内部模拟

每个能力对应一个子环境。WM要同时在所有子环境上表现好，
fitness = 5项能力的加权平均。

这是v24演化系统的测试床——架构沿用v21全器官，
但在这个更难的multi-task上跑。
"""
import random, math, copy, time

N_CHANNELS = 8; N_REGIONS = 9; DELAY = 3

# ============================================================
# 子环境1: 物体恒存性 (Object Permanence)
# ============================================================
class ObjectPermanenceEnv:
    """
    场景: 几个小球在各自轨道上运动。某些帧某些球会消失(遮挡)。
    任务: 球消失期间也要预测它的位置；重现时位置要连贯。
    
    信号编码:
      ch0: ball_A x位置 (sin运动)
      ch1: ball_A y位置
      ch2: ball_B x位置
      ch3: ball_B y位置
      ch4: ball_C x位置
      ch5: ball_C y位置
      ch6: 遮挡掩码 (哪些球当前可见)
      ch7: 时间戳
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.n_balls = 3
        self.period = [30, 45, 60]
        self.phase = [random.uniform(0, 2*math.pi) for _ in range(3)]
        self.amp = [0.8, 0.6, 0.4]
        self.t = 0
        self.occlusion_prob = 0.15
        self.occlusion_len = [5, 10]
    
    def reset(self):
        self.t = 0
        self.phase = [random.uniform(0, 2*math.pi) for _ in range(3)]
    
    def sense(self):
        t = self.t
        mask = 0
        vals = []
        for i in range(self.n_balls):
            px = self.amp[i] * math.sin(2*math.pi * t / self.period[i] + self.phase[i])
            py = self.amp[i] * math.cos(2*math.pi * t / self.period[i] * 0.7 + self.phase[i])
            # 遮挡逻辑
            occluded = (t % (self.period[i] * 2)) < random.randint(*self.occlusion_len)
            if occluded:
                mask |= (1 << i)
                vals.extend([0.0, 0.0])  # 遮挡时无信号
            else:
                vals.extend([px, py])
        vals.append(float(mask))
        vals.append(t % 100 / 100.0)
        return vals[:N_CHANNELS]
    
    def step(self):
        self.t += 1

# ============================================================
# 子环境2: 物理推理 (Physical Reasoning)
# ============================================================
class PhysicsEnv:
    """
    场景: 一个球在重力作用下从高处下落，碰到地面弹跳。
    任务: 预测球的y位置和速度，包括弹跳瞬间的反转。
    
    信号编码:
      ch0: y位置 (归一化-1~1)
      ch1: y速度
      ch2: 地面接触标志
      ch3: x位置 (水平匀速)
      ch4: x速度
      ch5: 能量 (0.5*v^2 + g*h)
      ch6: 反弹计数
      ch7: 时间戳
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.g = 0.015
        self.damping = 0.7
        self.reset()
    
    def reset(self):
        self.y = -0.8  # 起始高度
        self.vy = 0.0
        self.x = -0.9
        self.vx = 0.012
        self.t = 0
        self.bounces = 0
        self.prev_vy = 0.0
    
    def sense(self):
        contact = 1.0 if self.y >= 0.95 and self.prev_vy > 0 else 0.0
        energy = 0.5 * (self.vx**2 + self.vy**2) + self.g * (self.y + 1)
        return [
            max(-1, min(1, self.y)),
            max(-1, min(1, self.vy * 5)),
            contact,
            max(-1, min(1, self.x)),
            max(-1, min(1, self.vx * 10)),
            max(-1, min(1, energy)),
            self.bounces / 5.0,
            self.t % 100 / 100.0
        ]
    
    def step(self):
        self.prev_vy = self.vy
        self.vy += self.g  # 重力
        self.y += self.vy
        self.x += self.vx
        if self.x > 0.9 or self.x < -0.9: self.vx *= -1
        if self.y > 0.95 and self.vy > 0:  # 落地
            self.y = 0.95
            self.vy = -self.vy * self.damping
            if self.vy < -0.02:
                self.bounces += 1
        if self.y < -0.95:  # 顶部反弹
            self.y = -0.95
            self.vy = abs(self.vy) * 0.5
        self.t += 1

# ============================================================
# 子环境3: 空间关系 (Spatial Relations)
# ============================================================
class SpatialRelationEnv:
    """
    场景: 一个agent在2D平面上移动，周围有2个固定landmark。
    任务: 预测agent+landmark的相对位置，包括距离和角度。
    
    信号编码:
      ch0: agent_x
      ch1: agent_y
      ch2: dist_to_landmark_0
      ch3: angle_to_landmark_0
      ch4: dist_to_landmark_1
      ch5: angle_to_landmark_1
      ch6: landmark_0_visibility (偶尔不可见)
      ch7: 是否转弯标志
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.reset()
    
    def reset(self):
        self.ax = random.uniform(-0.5, 0.5)
        self.ay = random.uniform(-0.5, 0.5)
        self.dx = random.choice([-0.02, -0.01, 0.01, 0.02])
        self.dy = random.choice([-0.02, -0.01, 0.01, 0.02])
        self.lm = [[random.uniform(-0.8,0.8), random.uniform(-0.8,0.8)] for _ in range(2)]
        self.t = 0
        self.prev_ax = self.ax; self.prev_ay = self.ay
    
    def sense(self):
        dists = []
        angles = []
        vis = []
        for lm_pos in self.lm:
            d = math.sqrt((self.ax - lm_pos[0])**2 + (self.ay - lm_pos[1])**2)
            a = math.atan2(lm_pos[1] - self.ay, lm_pos[0] - self.ax) / math.pi
            v = 1.0 if (self.t % 40) < 35 else 0.0  # 偶尔不可见
            dists.append(min(1.0, d / 2))
            angles.append(a)
            vis.append(v)
        turning = 1.0 if (self.t % 20) == 0 else 0.0
        return [
            max(-1, min(1, self.ax)),
            max(-1, min(1, self.ay)),
            dists[0], angles[0], vis[0],
            dists[1], angles[1],
            turning
        ]
    
    def step(self):
        self.prev_ax = self.ax; self.prev_ay = self.ay
        # 偶尔转弯
        if self.t % 20 == 0 and random.random() < 0.5:
            self.dx, self.dy = -self.dy, self.dx
        self.ax += self.dx
        self.ay += self.dy
        if abs(self.ax) > 0.9: self.dx *= -1
        if abs(self.ay) > 0.9: self.dy *= -1
        self.ax = max(-0.95, min(0.95, self.ax))
        self.ay = max(-0.95, min(0.95, self.ay))
        self.t += 1

# ============================================================
# 子环境4: 多步生成预测 (Multi-step Generative Prediction)
# ============================================================
class MultiStepGenEnv:
    """
    场景: 一个复杂混沌系统 (Lorenz吸引子映射到8通道)。
    任务: 精确预测未来3帧 (不是1帧)。
    
    信号编码:
      ch0: lorenx (x/30)
      ch1: lorony (y/30)
      ch2: lorenz (z/30)+0.5
      ch3: dx/dt
      ch4: 李雅普诺夫距离(对初始条件敏感度)
      ch5-7: 正交变换
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.reset()
    
    def reset(self):
        self.x = random.uniform(-10, 10)
        self.y = random.uniform(-10, 10)
        self.z = random.uniform(10, 30)
        self.sigma = 10.0; self.rho = 28.0; self.beta = 8.0/3.0
        self.prev_x = self.x
        self.t = 0
    
    def sense(self):
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        v = math.sqrt(dx*dx + dy*dy + dz*dz) / 30.0
        return [
            max(-1, min(1, self.x / 30)),
            max(-1, min(1, self.y / 30)),
            max(-1, min(1, self.z / 50)),
            max(-1, min(1, v)),
            max(-1, min(1, abs(self.x - self.prev_x) / 5)),
            math.sin(self.x / 10), math.cos(self.y / 10),
            self.t % 100 / 100.0
        ]
    
    def step(self):
        self.prev_x = self.x
        dt = 0.02
        dx = self.sigma * (self.y - self.x)
        dy = self.x * (self.rho - self.z) - self.y
        dz = self.x * self.y - self.beta * self.z
        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt
        self.t += 1

# ============================================================
# 子环境5: 反事实推演 (Counterfactual)
# ============================================================
class CounterfactualEnv:
    """
    场景: 两个球在平面上运动，偶尔碰撞。
    任务: 给定"如果我没碰它"的反事实，预测不同的轨迹。
    
    信号编码:
      ch0: ball_A x
      ch1: ball_A y
      ch2: ball_B x
      ch3: ball_B y
      ch4: 碰撞标志
      ch5: 推力幅值 (实际操作)
      ch6: 反事实标志 (1=现实, 0=假设)
      ch7: 时间戳
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.reset()
    
    def reset(self):
        self.ax = random.uniform(-0.5, -0.2)
        self.ay = random.uniform(-0.5, -0.2)
        self.bx = random.uniform(0.2, 0.5)
        self.by = random.uniform(0.2, 0.5)
        self.avx = 0.015; self.avy = 0.01
        self.bvx = -0.01; self.bvy = -0.015
        self.t = 0
        self.cf_flag = random.choice([0, 1])
    
    def sense(self):
        d = math.sqrt((self.ax-self.bx)**2 + (self.ay-self.by)**2)
        collision = 1.0 if d < 0.1 else 0.0
        return [
            max(-1, min(1, self.ax)), max(-1, min(1, self.ay)),
            max(-1, min(1, self.bx)), max(-1, min(1, self.by)),
            collision,
            0.5 * (abs(self.avx) + abs(self.bvx)) * 10,
            float(self.cf_flag),
            self.t % 80 / 80.0
        ]
    
    def step(self):
        # 碰撞响应
        d = math.sqrt((self.ax-self.bx)**2 + (self.ay-self.by)**2)
        if d < 0.1 and d > 0.001:
            nx = (self.bx - self.ax) / d
            ny = (self.by - self.ay) / d
            self.avx += nx * 0.005; self.avy += ny * 0.005
            self.bvx -= nx * 0.005; self.bvy -= ny * 0.005
        self.ax += self.avx; self.ay += self.avy
        self.bx += self.bvx; self.by += self.bvy
        # 边界反弹
        for arr in [[self.ax, self.ay, self.avx, self.avy], [self.bx, self.by, self.bvx, self.bvy]]:
            if abs(arr[0]) > 0.9: arr[2] *= -1; arr[0] = max(-0.9, min(0.9, arr[0]))
            if abs(arr[1]) > 0.9: arr[3] *= -1; arr[1] = max(-0.9, min(0.9, arr[1]))
        # 偶尔注入反事实
        if self.t % 30 == 0:
            self.cf_flag = random.choice([0, 1])
        self.t += 1

# ============================================================
# 统一多能力评估
# ============================================================
class CapabilityTestbed:
    def __init__(self, seed=42):
        seed = hash((seed, 42)) % (2**31)
        self.envs = [
            ('object_permanence', ObjectPermanenceEnv(seed)),
            ('physics', PhysicsEnv(seed)),
            ('spatial_relation', SpatialRelationEnv(seed)),
            ('multi_step_gen', MultiStepGenEnv(seed)),
            ('counterfactual', CounterfactualEnv(seed)),
        ]
        self.weights = {
            'object_permanence': 0.25,
            'physics': 0.25,
            'spatial_relation': 0.2,
            'multi_step_gen': 0.15,
            'counterfactual': 0.15,
        }
    
    def reset(self):
        for _, env in self.envs:
            env.reset()
    
    def make_env_pool(self):
        """将5个环境打包成9个region（前5个region对应能力子环境）"""
        return self


def activation(x, leak=0.01):
    if x > 1.5: return 1.5 + leak * (x - 1.5)
    elif x < -1.5: return -1.5 + leak * (x + 1.5)
    else: return math.tanh(x)


# ============================================================
# v24WM — 带多能力评估的全器官WM
# ============================================================
class WM:
    def __init__(self, g):
        self.g = g; n = g['n']
        self.h = [0.0]*n; self.prev_h = [0.0]*n
        self.db_history = []; self.input_vec = []
        self.replay_buffer = []; self.REPLAY_CAP = 64
        self.homeo = {'hunger': 0.3, 'curiosity': 0.3, 'coherence': 0.3, 'engagement': 0.3}
        self.surprise_level = 0.0
        self.sens_adapt = [0.0] * N_CHANNELS; self.prev_sens_raw = None
        self.organ_activity = {'spike_count': 0, 'energy_use': 0.0, 'saturation': 0.0}
    
    def fwd_sensory(self, sens):
        g = self.g
        gain = g['sens_gain']; adapt_rate = g['sens_adapt_rate']
        result = list(sens)
        for c in range(N_CHANNELS):
            val = sens[c] * gain[c]
            if self.prev_sens_raw is not None:
                delta = abs(sens[c] - self.prev_sens_raw[c])
                self.sens_adapt[c] = min(0.85, self.sens_adapt[c]*(1-adapt_rate) + adapt_rate*(1.0-min(1.0,delta)))
                val *= (1.0 - 0.5 * self.sens_adapt[c])
            result[c] = val
        self.prev_sens_raw = list(sens)
        return result
    
    def fwd_association(self, x_raw):
        g = self.g; w = g['assoc_w']; thr = g['sparse_thr']; ext = []
        for i in range(N_CHANNELS): ext.append(x_raw[i])
        for i in range(N_CHANNELS): ext.append(math.tanh(x_raw[i]*w[0]))
        for i in range(N_CHANNELS):
            v = x_raw[i]
            if abs(v) > thr: ext.append(v*w[1])
            else: ext.append(0.0)
        for i in range(N_CHANNELS):
            c = sum(x_raw[j]*w[2]*0.1 for j in range(N_CHANNELS) if abs(i-j)==1)
            ext.append(c + x_raw[i]*w[2]*0.01)
        return ext
    
    def hippocampal_replay(self, event):
        buf = self.replay_buffer; buf.append(event)
        if len(buf) > self.REPLAY_CAP:
            buf.sort(key=lambda e: e['salience']); buf.pop(0)
    
    def hippocampal_learn(self):
        g = self.g
        if not self.replay_buffer: return
        hip_lr = g['hip_lr']; replay_n = g['hip_replay_n']
        events = sorted(self.replay_buffer, key=lambda e: e['salience'], reverse=True)[:replay_n]
        for ev in events:
            for i in range(min(len(self.h), len(g['pw']))):
                for c in range(N_CHANNELS):
                    g['pw'][i][c] += hip_lr * ev['salience'] * self.h[i] * 0.01
                    g['pw'][i][c] = max(-3, min(3, g['pw'][i][c]))
    
    def amygdala_surprise(self, mse):
        g = self.g; thr = g['amy_thr']
        prev = self.surprise_level
        self.surprise_level = max(0.0, (mse - thr) / max(0.01, 1.0 - thr))
        return self.surprise_level, max(0, self.surprise_level - prev)
    
    def protein_fold(self):
        g = self.g; n = g['n']; fold_lr = g['fold_lr']
        rw = g['rw']; ph = self.prev_h; h = self.h
        lk = g.get('leak', [0.01]*n); thr = g.get('fold_threshold', 0.01)
        for k in range(n):
            rwk = rw[k]; predicted = 0.0
            for i in range(n): predicted += rwk[i] * ph[i]
            if abs(predicted) > thr:
                predicted = activation(predicted, lk[k] if k < len(lk) else 0.01)
                err = h[k] - predicted
                for i in range(n):
                    rwk[i] += fold_lr * ph[i] * err
                    if rwk[i] > 1.2: rwk[i] = 1.2
                    elif rwk[i] < -1.2: rwk[i] = -1.2
    
    def learn(self, pred, actual):
        g = self.g; n = g['n']; lr = g['lr']
        errs = [actual[j]-pred[j] for j in range(N_CHANNELS)]
        mse = sum(e*e for e in errs) / N_CHANNELS
        surprise, edge = self.amygdala_surprise(mse)
        if surprise > 0.01:
            self.hippocampal_replay({'target': actual[:], 'salience': surprise})
        pw = g['pw']; pb = g['pb']; h = self.h
        for i in range(min(n, len(pw))):
            pwi = pw[i]; hi = h[i]
            for c in range(N_CHANNELS):
                pwi[c] += lr * hi * errs[c]
                if pwi[c] > 3: pwi[c] = 3
                elif pwi[c] < -3: pwi[c] = -3
        for c in range(N_CHANNELS):
            pb[c] += lr * 1.5 * errs[c]
            if pb[c] > 5: pb[c] = 5
            elif pb[c] < -5: pb[c] = -5
        x = self.input_vec; skip = g['skip']
        for j in range(min(len(skip), len(x))):
            skj = skip[j]; xj = x[j]
            for c in range(N_CHANNELS):
                skj[c] += lr * 0.5 * xj * errs[c]
                if skj[c] > 2: skj[c] = 2
                elif skj[c] < -2: skj[c] = -2
        self.protein_fold()
        self.organ_activity['spike_count'] = sum(1 for v in h if abs(v) > 0.5)
        self.organ_activity['saturation'] = sum(1 for v in h if abs(v) > 0.98)/max(1,n)
        return mse
    
    def build_input(self, sens):
        self.db_history.append(list(sens[:N_CHANNELS]))
        if len(self.db_history) > DELAY: self.db_history.pop(0)
        while len(self.db_history) < DELAY: self.db_history.insert(0, [0.0]*N_CHANNELS)
        result = list(sens[:N_CHANNELS])
        for frame in self.db_history: result.extend(frame)
        self.input_vec = result[:]
        return result
    
    def fwd(self, sens, update=True):
        g = self.g; n = g['n']; dec = g['dt'][0]
        sens_proc = self.fwd_sensory(list(sens[:N_CHANNELS]))
        extended = self.fwd_association(sens_proc)
        x = self.build_input(extended[:N_CHANNELS])
        sw = g['sw']; rw = g['rw']; lk = g.get('leak', [0.01]*n)
        h = self.h; nh = [0.0]*n
        sat_p = g.get('sat_prevent', 0.9)
        for i in range(n):
            a = 0.0
            for j in range(min(len(x), len(sw))): a += sw[j][i] * x[j]
            rwi = rw[i]
            for k in range(n): a += rwi[k] * h[k]
            nh[i] = activation(a * sat_p, lk[i] if i < len(lk) else 0.01)
        if update:
            for i in range(n): self.h[i] = (1-dec)*nh[i] + dec*self.h[i]
        pred = [0.0]*N_CHANNELS
        pw = g['pw']; pb = g['pb']; skip = g['skip']; hv = self.h
        for i in range(min(n, len(pw))):
            pwi = pw[i]; hi = hv[i]
            for c in range(N_CHANNELS): pred[c] += hi*(pwi[c]+pb[c])
        for j in range(min(len(skip), len(x))):
            skj = skip[j]; xj = x[j]
            for c in range(N_CHANNELS):
                if c == 3 and 'ch3_boost' in g:
                    pred[c] += skj[c] * xj * g['ch3_boost']
                else:
                    pred[c] += skj[c] * xj
        if update: self.prev_h = self.h[:]
        return pred
    
    def reset(self):
        self.h = [0.0]*len(self.h); self.prev_h = [0.0]*len(self.prev_h)
        self.db_history = []; self.input_vec = []
        self.replay_buffer = []
        self.homeo = {'hunger': 0.3, 'curiosity': 0.3, 'coherence': 0.3, 'engagement': 0.3}
        self.surprise_level = 0.0
        self.sens_adapt = [0.0]*N_CHANNELS; self.prev_sens_raw = None


# ============================================================
# 演化引擎
# ============================================================
class Org:
    def __init__(self, gn=None):
        self.g = gn or self.rg(); self.wm = WM(self.g)
    
    def rg(self):
        n = random.randint(12, 40); ins = N_CHANNELS*(1+DELAY)
        g = {
            'n': n,
            'sw': [[random.uniform(-0.3,0.3) for _ in range(n)] for __ in range(ins)],
            'pw': [[random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)] for __ in range(n)],
            'rw': [[random.uniform(-0.03,0.03) for _ in range(n)] for __ in range(n)],
            'skip': [[random.uniform(-0.2,0.2) for _ in range(N_CHANNELS)] for __ in range(ins)],
            'pb': [random.uniform(-0.3,0.3) for _ in range(N_CHANNELS)],
            'dt': [random.uniform(0.5,0.9)], 'lr': random.uniform(0.04,0.12),
            'fold_lr': random.uniform(0.01,0.06), 'leak': [random.uniform(0.005,0.05) for _ in range(n)],
            'sat_prevent': random.uniform(0.8,0.95), 'adapt_cap': random.uniform(0.7,0.9),
            'ch3_boost': random.uniform(1.2,2.0),
            'sens_gain': [random.uniform(0.5,2.0) for _ in range(N_CHANNELS)],
            'sens_adapt_rate': random.uniform(0.01,0.1),
            'assoc_w': [random.uniform(0.3,1.5) for _ in range(3)], 'sparse_thr': random.uniform(0.05,0.5),
            'hip_lr': random.uniform(0.001,0.02), 'hip_replay_n': random.randint(2,8),
            'amy_thr': random.uniform(0.02,0.15), 'pfc_gate_thr': random.uniform(0.05,0.3),
            'pfc_gate_lr': random.uniform(0.01,0.1), 'bg_explore_rate': random.uniform(0.05,0.2),
            'bg_noise': random.uniform(0.01,0.1), 'hypo_drive': [random.uniform(0.1,0.5) for _ in range(4)],
            'hypo_fill': random.uniform(0.005,0.03), 'fold_threshold': random.uniform(0.005,0.05),
        }
        for i in range(n): g['rw'][i][i] = random.uniform(0.4,1.0)
        return g


class Engine:
    def __init__(self, n_org=20, n_gen=200, seed=42, ticks_per_env=100):
        self.n_org = n_org; self.n_gen = n_gen
        self.ticks_per_env = ticks_per_env
        self.pop = [Org() for _ in range(n_org)]
        self.best_ever = 0.0; self.best_g = None; self.log = []
        random.seed(seed)
        self.testbed = CapabilityTestbed(seed)
    
    def eval_org(self, org):
        """在5个子环境上分别评估"""
        scores = {}
        org.wm.reset()
        
        for env_name, env in self.testbed.envs:
            env.reset()
            # warmup
            for _ in range(10):
                s = env.sense(); org.wm.fwd(s, update=True); env.step()
            total_mse = 0.0
            for t in range(self.ticks_per_env):
                s = env.sense(); pred = org.wm.fwd(s, update=True); env.step()
                ns = env.sense()  # 下一帧真实值作为target
                mse = sum((ns[j]-pred[j])**2 for j in range(N_CHANNELS))/N_CHANNELS
                org.wm.learn(pred, ns); total_mse += mse
                env.step()
            scores[env_name] = 1.0 / (1.0 + total_mse / self.ticks_per_env)
        
        # 加权总分
        weighted = sum(scores[k] * self.testbed.weights[k] for k in scores)
        return weighted, scores
    
    def xover(self, a, b):
        n = max(a['n'],b['n']); ins = N_CHANNELS*(1+DELAY)
        ch = {'n':n,'sw':[],'pw':[],'rw':[],'skip':[],
              'pb':a['pb'][:] if random.random()<0.5 else b['pb'][:],
              'dt':a['dt'][:] if random.random()<0.5 else b['dt'][:],
              'lr':a['lr'] if random.random()<0.5 else b['lr'],
              'fold_lr':a['fold_lr'] if random.random()<0.5 else b['fold_lr']}
        for k in ['sat_prevent','adapt_cap','ch3_boost','sens_adapt_rate','sparse_thr',
                  'hip_lr','hip_replay_n','amy_thr','pfc_gate_thr','pfc_gate_lr',
                  'bg_explore_rate','bg_noise','hypo_fill','fold_threshold']:
            ch[k] = a.get(k,0.5) if random.random()<0.5 else b.get(k,0.5)
        ch['sens_gain'] = [a['sens_gain'][i] if random.random()<0.5 else b['sens_gain'][i] for i in range(N_CHANNELS)]
        ch['assoc_w'] = [a['assoc_w'][i] if random.random()<0.5 else b['assoc_w'][i] for i in range(3)]
        ch['hypo_drive'] = [a['hypo_drive'][i] if random.random()<0.5 else b['hypo_drive'][i] for i in range(4)]
        for j in range(ins):
            src = a if random.random()<0.5 else b
            ch['sw'].append([src['sw'][min(j,len(src['sw'])-1)][min(i,src['n']-1)] if min(j,len(src['sw'])-1)>=0 else 0.0 for i in range(n)])
        for i in range(n):
            src = a if random.random()<0.5 else b; si = min(i, src['n']-1)
            ch['pw'].append(src['pw'][si][:N_CHANNELS] if si < len(src['pw']) else [0.0]*N_CHANNELS)
        for i in range(n):
            src = a if random.random()<0.5 else b; si = min(i, src['n']-1)
            ch['rw'].append([src['rw'][si][min(k,src['n']-1)] if si < len(src['rw']) and min(k,src['n']-1) < len(src['rw'][si]) else 0.0 for k in range(n)])
        for j in range(ins):
            src = a if random.random()<0.5 else b
            ch['skip'].append(src['skip'][min(j,len(src['skip'])-1)][:N_CHANNELS])
        ch['leak'] = [a['leak'][i] if i<len(a['leak']) and random.random()<0.5 else b['leak'][i] if i<len(b['leak']) else 0.02 for i in range(n)]
        ch['hip_replay_n'] = a['hip_replay_n'] if random.random()<0.5 else b['hip_replay_n']
        return ch
    
    def mutate(self, g, rate=0.03):
        n = g['n']
        for j in range(len(g['sw'])):
            for i in range(len(g['sw'][j])):
                if random.random()<rate:
                    g['sw'][j][i]+=random.gauss(0,0.05); g['sw'][j][i]=max(-3,min(3,g['sw'][j][i]))
        for i in range(min(n,len(g['pw']))):
            for c in range(N_CHANNELS):
                if random.random()<rate:
                    g['pw'][i][c]+=random.gauss(0,0.05); g['pw'][i][c]=max(-3,min(3,g['pw'][i][c]))
        for i in range(min(n,len(g['rw']))):
            for k in range(min(n,len(g['rw'][i]))):
                if random.random()<rate*0.7:
                    g['rw'][i][k]+=random.gauss(0,0.03); g['rw'][i][k]=max(-1.5,min(1.5,g['rw'][i][k]))
        for j in range(len(g['skip'])):
            for c in range(N_CHANNELS):
                if random.random()<rate:
                    g['skip'][j][c]+=random.gauss(0,0.05); g['skip'][j][c]=max(-2,min(2,g['skip'][j][c]))
        # 器官参数变异
        mut_params = {
            'sat_prevent': (0.7,0.99,0.03), 'adapt_cap': (0.5,0.95,0.05), 'ch3_boost': (1.0,3.0,0.2),
            'sens_adapt_rate': (0.001,0.2,0.02), 'hip_lr': (0.0005,0.05,0.005),
            'amy_thr': (0.01,0.3,0.02), 'bg_explore_rate': (0.02,0.4,0.03),
            'fold_threshold': (0.001,0.1,0.01), 'lr': (0.02,0.2,0.01), 'fold_lr': (0.005,0.08,0.008),
        }
        for k, (lo,hi,std) in mut_params.items():
            if random.random()<0.25:
                g[k] = max(lo, min(hi, g[k] + random.gauss(0,std)))
        if random.random()<0.2:
            idx = random.randint(0,N_CHANNELS-1)
            g['sens_gain'][idx] = max(0.1,min(3.0,g['sens_gain'][idx]+random.gauss(0,0.1)))
        if random.random()<0.2:
            idx = random.randint(0,2)
            g['assoc_w'][idx] = max(0.1,min(2.0,g['assoc_w'][idx]+random.gauss(0,0.1)))
        if random.random()<0.25:
            g['hip_replay_n'] = max(1,min(12,g['hip_replay_n']+random.choice([-1,1])))
        return g
    
    def evolve(self):
        st = time.time()
        for gen in range(self.n_gen):
            all_scores = []
            for o in self.pop:
                weighted, detail = self.eval_org(o)
                all_scores.append((weighted, detail))
            fits = [s[0] for s in all_scores]
            bi = fits.index(max(fits)); bf = fits[bi]; af = sum(fits)/len(fits)
            if bf > self.best_ever:
                self.best_ever = bf; self.best_g = copy.deepcopy(self.pop[bi].g)
            if gen%20==0 or gen==self.n_gen-1:
                ng = self.best_g['n'] if self.best_g else 0
                detail_str = ' '.join(f'{k[:3]}={all_scores[bi][1][k]:.3f}' for k in all_scores[bi][1])
                print(f"Gen {gen:4d} {time.time()-st:6.1f}s bf={bf:.4f} af={af:.4f} best={self.best_ever:.4f} n={ng}")
                print(f"          {detail_str}")
                self.log.append((gen, bf, af, self.best_ever, dict(all_scores[bi][1])))
            if gen == self.n_gen-1: break
            idx = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
            nx = [Org(copy.deepcopy(self.pop[idx[i]].g)) for i in range(min(4,len(self.pop)))]
            while len(nx) < self.n_org:
                t1 = random.sample(range(len(self.pop)), min(5,len(self.pop))); p1 = max(t1, key=lambda i: fits[i])
                t2 = random.sample(range(len(self.pop)), min(5,len(self.pop))); p2 = max(t2, key=lambda i: fits[i])
                if random.random()<0.2 and len(self.pop)>5:
                    wild = int(random.random()**2*len(self.pop)); child = self.xover(self.pop[wild].g, self.pop[idx[0]].g)
                else: child = self.xover(self.pop[p1].g, self.pop[p2].g)
                self.mutate(child, rate=0.03); nx.append(Org(child))
            self.pop = nx
        return self.best_ever

if __name__ == '__main__':
    print("="*70)
    print("v24_capability_testbed — 多能力测试床")
    print("测试: 物体恒存+物理推理+空间关系+多步生成+反事实")
    print("="*70)
    eng = Engine(n_org=20, n_gen=200, seed=42, ticks_per_env=100)
    result = eng.evolve()
    print(f"\n最终最佳: {result:.4f}")
    
    if eng.best_g:
        g = eng.best_g
        print(f"结构: n={g['n']} lr={g['lr']:.4f} fold={g['fold_lr']:.4f} sat_p={g.get('sat_prevent',0.9):.3f}")
    
    if eng.log:
        final_detail = eng.log[-1][4]
        print("\n各能力分:")
        for k, v in final_detail.items():
            w = eng.testbed.weights[k]
            print(f"  {k:20s}: fitness={v:.4f} (weight={w})")
