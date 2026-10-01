# yinyang_evo_v9.py
# 阴阳扣数字生命系统 · v9 — Intrinsic Motivation / "Autonomous Drive"
#
# 核心改动：v8 的适合度完全由外部定义（预测误差→能量→繁殖）
# v9 增加"内在动机"：学习本身的愉悦感。让洛神"自己想要"变强。
#
# 四项内在灵感：
#   1. CURIOUS (好奇心)：对预测不准的感官状态给予探索奖励
#      驱动：主动寻找自己没见过的环境区域
#   2. MASTERY (精通)：对自身预测误差下降给予奖励
#      驱动：不满足于比随机好，要比昨天的自己更好
#   3. AGENCY (因果欲)：行动后环境产生了预期变化时给予奖励
#      驱动：想搞清楚"我做了什么→世界变了什么"
#   4. NOVELTY (新奇)：对与自身历史完全不同的感官状态给予奖励
#      驱动：对重复无聊的事产生排斥（boredom penalty）
#
# 理论上：不做外部给目标，这四个内在驱动力应该让洛神
# "自己想要"扩张感知能力、理解因果关系、探索环境。

import random, math, json, os, pickle, hashlib
from copy import deepcopy
from collections import deque, Counter

MAX_NODES = 12
N_CHANNELS = 8
N_ACTIONS = 4
CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']

# ============================================================
# 环境（v8 完全保留）
# ============================================================

class Environment:
    def __init__(self, seed=None, mode='training', difficulty=1.0):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.mode = mode
        self.difficulty = difficulty
        self.season = 0
        self.substep = 0
        self.light = 50.0; self.light_r = 0.0
        self.sound_freq = 440.0; self.sound_amp = 30.0
        self.pressure = 0.0; self.vibration = 0.0
        self.smell = [0.0]*5; self.taste = [0.0]*5
        self.temperature = 20.0; self.accel = [0.0, 0.0, 9.8]

    def step(self, action_code=1):
        self.t += 1
        self.substep += 1
        if self.substep > max(1, int(150 / self.difficulty)):
            self.substep = 0
            self.season = (self.season + 1) % 4
        s = self.season

        if action_code == 0:
            self.light += random.uniform(0.3, 1.0)
            self.sound_amp += random.uniform(0.05, 0.2)
        elif action_code == 2:
            self.light *= 0.97; self.pressure *= 0.92
        elif action_code == 3:
            self.temperature += random.uniform(-0.5, 0.5)
            self.pressure += random.uniform(0.05, 0.2)

        diff = self.difficulty
        base_light = 50 + 25 * math.sin(self.t * 0.02 + s * math.pi / 2)
        noise_light = random.uniform(-2, 2) * diff

        if self.mode == 'training':
            if s == 0:
                self.light += 0.08 * (base_light - self.light) + noise_light * 0.5
                self.sound_amp += random.uniform(-0.3, 0.1) * diff
                self.pressure += random.uniform(-0.05, 0.05) * diff
                if self.substep % max(1, int(20 / diff)) == 0:
                    self.sound_amp += random.uniform(3, 6) * diff
            elif s == 1:
                self.light += 0.06 * (30 - self.light) + noise_light
                self.sound_amp += random.uniform(-0.2, 0.6) * diff
                if random.random() < 0.02 * diff:
                    self.sound_amp += random.uniform(5, 12) * diff
            elif s == 2:
                self.temperature += 0.06 * (28 - self.temperature) + random.uniform(-0.05, 0.05) * diff
                self.pressure += 0.05 * (3 - self.pressure)
                temp_effect = (self.temperature - 20) * 0.3
                for i in range(5): self.smell[i] += temp_effect * 0.1 * diff
            else:
                self.temperature += 0.06 * (17 - self.temperature)
                self.pressure += 0.05 * (15 - self.pressure) + random.uniform(-0.1, 0.1) * diff
                if random.random() < 0.08 * diff:
                    self.accel[0] += random.uniform(-1.5, 1.5) * diff
                    self.accel[1] += random.uniform(-1.5, 1.5) * diff
        else:
            if s == 0:
                v = 10 * math.sin(self.t * 0.03) + 5 * math.sin(self.t * 0.11)
                self.light += 0.1 * (50 + v - self.light) + random.uniform(-1, 1) * diff
                self.pressure += random.uniform(-0.1, 0.15) * diff
            elif s == 1:
                x = (self.pressure + 10) / 20
                x = 3.7 * x * (1 - x)
                self.pressure = max(0, min(20, x * 20))
                self.sound_amp += random.uniform(-0.5, 0.8) * diff
            elif s == 2:
                if self.substep == 50: self.light = 100 - self.light
                self.temperature += random.uniform(-0.08, 0.08) * diff
                self.sound_freq += random.uniform(-15, 15)
            else:
                self.light += random.uniform(-3, 3) * diff
                pulse = 3.0 if self.substep % 30 < 3 else 0
                self.sound_amp += random.uniform(-0.8, 0.8) * diff + pulse
                self.accel[2] = 9.8 + 2 * math.sin(self.t * 0.2)

        self.light = max(0.0, min(100.0, self.light))
        self.light_r = max(-1.0, min(1.0, self.light_r + random.uniform(-0.01, 0.01)))
        self.sound_freq = max(20.0, min(20000.0, self.sound_freq))
        self.sound_amp = max(0.0, min(120.0, self.sound_amp * 0.99))
        self.pressure = max(0.0, min(20.0, self.pressure))
        self.vibration = max(-2.0, min(2.0, self.vibration * 0.9 + random.uniform(-0.1, 0.1)))
        for i in range(5):
            self.smell[i] *= 0.93
            self.smell[i] = max(0.0, min(10.0, self.smell[i]))
            if random.random() < 0.02:
                self.smell[i] += random.uniform(0, 2)
            self.taste[i] *= 0.985
            if random.random() < 0.04:
                self.taste[i] += random.uniform(-0.3, 0.3)
            self.taste[i] = max(-5.0, min(5.0, self.taste[i]))
        self.temperature = max(15.0, min(35.0, self.temperature))
        for k in range(3):
            self.accel[k] = max(-2.0, min(12.0, self.accel[k]))

    def get_sensory(self):
        return [
            (self.light / 50.0) - 1.0,
            (self.sound_amp / 60.0) - 0.5 + (math.log(max(1.0, self.sound_freq)) - 6) * 0.15,
            (self.pressure / 10.0) - 0.5,
            (sum(self.smell) / 25.0) - 0.5,
            (sum(self.taste) / 10.0),
            (self.temperature - 25.0) / 10.0,
            (abs(self.accel[2] - 9.8) / 5.0),
            0.0
        ]


# ============================================================
# WorldModel（v8 完全保留）
# ============================================================

def resize_genome(genome, new_n):
    g = deepcopy(genome)
    old_n = g['params']['n_nodes']
    new_sensory = []
    for i in range(N_CHANNELS):
        row = list(g['sensory_w'][i][:min(old_n, new_n)])
        while len(row) < new_n: row.append(random.uniform(-0.3, 0.3))
        new_sensory.append(row)
    g['sensory_w'] = new_sensory
    new_predict = []
    for i in range(new_n):
        if i < old_n: new_predict.append(list(g['predict_w'][i]))
        else: new_predict.append([random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)])
    g['predict_w'] = new_predict
    new_rec = [[0.0]*new_n for _ in range(new_n)]
    for i in range(min(old_n, new_n)):
        for j in range(min(old_n, new_n)): new_rec[i][j] = g['recurrent_w'][i][j]
    for i in range(old_n, new_n):
        for j in range(old_n, new_n):
            if i != j: new_rec[i][j] = random.uniform(-0.1, 0.1)
    g['recurrent_w'] = new_rec
    new_action = []
    for a in range(N_ACTIONS):
        row = list(g['action_w'][a][:min(old_n, new_n)])
        while len(row) < new_n: row.append(random.uniform(-0.2, 0.2))
        new_action.append(row)
    g['action_w'] = new_action
    g['params']['n_nodes'] = new_n
    return g


class WorldModel:
    def __init__(self, genome):
        self.params = genome['params']
        n = self.params['n_nodes']
        self.nodes = [0.0] * n
        self.last_prediction = [0.0] * N_CHANNELS
        self.sensory_w = genome['sensory_w']
        self.predict_w = genome['predict_w']
        self.recurrent_w = genome['recurrent_w']
        self.action_w = genome['action_w']
        self.action_bias = genome['action_bias']
        self.prediction_errors = deque(maxlen=80)
        self.cumulative_mse = 0.0
        self.n_predictions = 0

    def forward(self, sensory_vec):
        n = len(self.nodes)
        decay = self.params['wm_decay']
        predicted_next = []
        for i in range(N_CHANNELS):
            pred = 0.0
            for j in range(n): pred += self.predict_w[j][i] * self.nodes[j]
            predicted_next.append(math.tanh(pred * self.params['pred_gain']))
        new_nodes = []
        for j in range(n):
            val = self.nodes[j] * decay
            for i in range(N_CHANNELS): val += self.sensory_w[i][j] * sensory_vec[i] * self.params['sensory_gain']
            for k in range(n): val += self.recurrent_w[j][k] * self.nodes[k]
            if self.last_prediction and len(self.last_prediction) == N_CHANNELS:
                for i in range(N_CHANNELS):
                    err = sensory_vec[i] - self.last_prediction[i]
                    val += err * self.sensory_w[i][j] * 0.05
            new_nodes.append(math.tanh(val))
        self.nodes = new_nodes
        self.last_prediction = predicted_next[:]
        return predicted_next[:], new_nodes[:]

    def choose_action(self, bias=None):
        n = len(self.nodes)
        logits = []
        for a in range(N_ACTIONS):
            logit = self.action_bias[a]
            for j in range(n): logit += self.action_w[a][j] * self.nodes[j]
            if bias and a in bias: logit += bias[a]
            logits.append(logit)
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_e = sum(exps)
        if sum_e == 0: return random.randrange(N_ACTIONS)
        probs = [e / sum_e for e in exps]
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative: return i
        return N_ACTIONS - 1

    def forward_from_nodes(self, nodes, sensory_vec):
        n = len(nodes)
        decay = self.params['wm_decay']
        predicted_next = []
        for i in range(N_CHANNELS):
            pred = 0.0
            for j in range(min(n, len(self.predict_w))): pred += self.predict_w[j][i] * nodes[j]
            predicted_next.append(math.tanh(pred * self.params['pred_gain']))
        new_nodes = []
        for j in range(n):
            val = nodes[j] * decay
            for i in range(N_CHANNELS):
                if j < len(self.sensory_w[i]): val += self.sensory_w[i][j] * sensory_vec[i] * self.params['sensory_gain']
            for k in range(n):
                if k < len(self.recurrent_w[j]) and j < len(self.recurrent_w): val += self.recurrent_w[j][k] * nodes[k]
            if self.last_prediction and len(self.last_prediction) == N_CHANNELS:
                for i in range(N_CHANNELS):
                    if j < len(self.sensory_w[i]):
                        err = sensory_vec[i] - self.last_prediction[i]
                        val += err * self.sensory_w[i][j] * 0.03
            new_nodes.append(math.tanh(val))
        return predicted_next, new_nodes

    def learn(self, actual_next, predicted_next, rate):
        n = len(self.nodes)
        errors = [actual_next[i] - predicted_next[i] for i in range(N_CHANNELS)]
        mse = sum(e*e for e in errors) / N_CHANNELS
        self.prediction_errors.append(mse)
        self.cumulative_mse += mse
        self.n_predictions += 1
        for i in range(N_CHANNELS):
            for j in range(n):
                delta = rate * errors[i] * self.nodes[j] * 0.01
                self.predict_w[j][i] += delta
                self.predict_w[j][i] = max(-5.0, min(5.0, self.predict_w[j][i]))
        for j in range(n):
            for i in range(N_CHANNELS):
                delta = rate * self.nodes[j] * actual_next[i] * 0.001
                self.sensory_w[i][j] += delta
                self.sensory_w[i][j] = max(-5.0, min(5.0, self.sensory_w[i][j]))


# ============================================================
# 内在动机模块 v9 新增
# ============================================================

class IntrinsicMotivation:
    """
    四项内在奖励 + 整体 boredom 检测。
    洛神不是因为外部说"你做得好"而高兴，而是"我能预测这个""我学到了新东西""我的行动
    确实改变了世界"——这些事情本身就让它爽。
    """
    def __init__(self):
        # 指数移动平均（跟踪自身能力基线）
        self.ema_error = 0.05  # 预测误差的 EMA
        self.ema_error_fast = 0.05  # 快速版本（检测即时变化）
        # 感官历史缓冲（用于新奇度估计）
        self.sensory_history = deque(maxlen=400)
        # 行动-效果记忆（因果学习）
        self.action_outcomes = {}
        for a in range(N_ACTIONS):
            self.action_outcomes[a] = deque(maxlen=50)
        # 内在能量（——独立于生存能量，只做自我奖励）
        self.drive_energy = 0.0
        # boredom 累计（每 tick + boredom_penalty if too predictable）
        self.boredom = 0.0

    def update_and_reward(self, sensory_before, sensory_after, predicted, action_taken, actual_mse):
        """学习周期结束时调用，返回内在能量奖励"""
        r = 0.0
        N = N_CHANNELS

        # EMA 更新
        alpha_slow, alpha_fast = 0.02, 0.1
        self.ema_error = (1 - alpha_slow) * self.ema_error + alpha_slow * actual_mse
        self.ema_error_fast = (1 - alpha_fast) * self.ema_error_fast + alpha_fast * actual_mse

        # 1. CURIOUS：当前误差快于基线 → "我在这里不够懂" → 探索 reward
        #    当前误差快 → 这个环境区域对我来说"有信息" → 值得深入了解
        if self.ema_error_fast > self.ema_error * 1.2:
            reward_curious = min(0.05, (self.ema_error_fast - self.ema_error) * 0.5)  # 最大 0.3
            r += reward_curious

        # 2. MASTERY：自身能力进步 → "我变强了" → 精通 reward
        #    EMA 下降 = 平均误差减小 = 变强
        #    注：用一个逆指标（大 → 好），所以 reward = baseline - current
        mastery_signal = max(0.0, 0.05 - actual_mse)  # 误差小于 0.05 时给正向奖励
        if mastery_signal > 0:
            # 误差越低奖励越大（对数）
            mastery_reward = min(0.05, (0.05 - actual_mse) * 0.2)
            r += mastery_reward

        # 3. AGENCY：行动有因果效应 → "我的行动改变了世界" → 控制感 reward
        #    衡量：行动后感官状态的变化量 vs 预测变化量
        if sensory_before is not None:
            actual_delta = sum((sensory_after[i] - sensory_before[i])**2 for i in range(N))
            expected_delta = sum((predicted[i] - sensory_before[i])**2 for i in range(N))
            # 行动产生了预期之外的变化 = 可能有agent被发现，给予奖励
            # 或者，行动产生了精确预期的变化 = 因果模型正确
            if expected_delta > 0.01:
                # 实际变化与预期变化匹配 → "我搞清楚了世界"
                match = 1.0 / (1.0 + abs(actual_delta - expected_delta))
                agency_reward = match * 0.03
                r += agency_reward
                self.action_outcomes[action_taken].append((expected_delta, actual_delta, match))

        # 4. NOVELTY：这个感官状态以前没见过（距离历史很远）→ 新奇 reward
        if self.sensory_history:
            dists = []
            for past_s in self.sensory_history:
                d = math.sqrt(sum((sensory_after[i] - past_s[i])**2 for i in range(N)))
                dists.append(d)
                if len(dists) > 50: break  # 限制计算量
            if dists:
                min_dist = min(dists)
                max_dist = max(dists) if max(dists) > 0 else 1.0
                novelty = min_dist / max(1e-3, max_dist)  # 越远离历史越新奇
                novelty_reward = min(0.05, novelty * 0.08)
                r += novelty_reward
        self.sensory_history.append(sensory_after[:])

        # BOREDOM：连续可预测 → 负奖励，促使探索或换策略
        if actual_mse < 0.002 and self.ema_error_fast < 0.002:
            self.boredom += 0.001  # 每次太可预测就积累无聊度
            if self.boredom > 0.05:
                r -= 0.02  # 足够无聊则扣奖励
                self.boredom = 0.0  # 重置，等下次无聊积累
        else:
            self.boredom *= 0.95  # 逐渐衰减

        self.drive_energy = self.drive_energy * 0.99 + r
        self.drive_energy = max(-5.0, min(10.0, self.drive_energy))  # 有界

                # 行动 bias 已在 plan() 中直接计算，此处只返回 reward
        return r


# ============================================================
# Organism v9
# ============================================================

class Organism:
    CHANNELS = CHANNELS

    def __init__(self, genome=None, lineage_id=None):
        self.genome = genome if genome else self.random_genome()
        self.wm = WorldModel(self.genome)
        self.im = IntrinsicMotivation()  # v9 新增：内在动机模块
        self.senses_prev = None
        self.senses_before_action = None  # v9：记录行动前的感官（用于因果学习）
        self.state = [0.0] * N_CHANNELS
        self.state_history = []
        self.action_history = []
        self.alive = True
        self.age = 0
        self.energy = 8.0  # 生存能量（外部）
        self.lineage_id = lineage_id or self._gen_id()
        self.parents = []
        self.behavior_trace = []
        self._predicted_next = None
        self._action_chosen = None
        self.intrinsic_log = deque(maxlen=50)  # 记录内在动机变化

    def wm_forward_from_nodes(self, nodes, sensory_vec):
        return self.wm.forward_from_nodes(nodes, sensory_vec)

    def _gen_id(self):
        return hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]

    @staticmethod
    def random_genome():
        n = random.randint(3, MAX_NODES)
        return {
            'sensory_w': [[random.uniform(-0.3, 0.3) for _ in range(n)] for _ in range(N_CHANNELS)],
            'predict_w': [[random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)] for _ in range(n)],
            'recurrent_w': [[random.uniform(-0.1, 0.1) if i != j else 0.0 for j in range(n)] for i in range(n)],
            'action_w': [[random.uniform(-0.2, 0.2) for _ in range(n)] for _ in range(N_ACTIONS)],
            'action_bias': [random.uniform(-0.1, 0.1) for _ in range(N_ACTIONS)],
            'params': {
                'n_nodes': n,
                'wm_decay': random.uniform(0.7, 0.95),
                'sensory_gain': random.uniform(0.3, 2.0),
                'pred_gain': random.uniform(0.5, 2.0),
                'learning_rate': random.uniform(0.01, 0.3),
                'metabolic': random.uniform(0.02, 0.12),
            }
        }

    def plan(self, sensory):
        if self.senses_prev is not None:
            proprio = math.sqrt(sum((sensory[i] - self.senses_prev[i])**2 for i in range(N_CHANNELS))) / N_CHANNELS
            sensory[N_CHANNELS - 1] = proprio
        else:
            sensory[N_CHANNELS - 1] = 0.0

        # v9 新增：记录行动前的感官
        self.senses_before_action = sensory[:]
        self.senses_prev = sensory[:]
        self.state = sensory[:]
        self.state_history.append(sensory[:])
        if len(self.state_history) > 300: self.state_history = self.state_history[-300:]

        predicted, nodes = self.wm.forward(sensory)
        self._predicted_next = predicted

        # v9：根据内在动机偏差选择行动（但不计入奖励，奖励只在 learn 阶段计）
        action_bias = None
        if self.im.ema_error_fast > self.im.ema_error * 1.3:
            action_bias = {3: 0.3}  # 好奇 → 探索高风险
        elif self.im.boredom > 0.03:
            action_bias = {0: 0.2, 2: 0.2}  # 无聊 → 换策略
        
        action_idx = self.wm.choose_action(bias=action_bias)  # v9
        self._action_chosen = action_idx
        self.action_history.append(action_idx)
        if len(self.action_history) > 200: self.action_history = self.action_history[-200:]

        return predicted, action_idx

    def learn_and_update(self, actual_next, action_code):
        if not self.alive: return
        predicted = self._predicted_next
        if predicted is None: return

        self.wm.learn(actual_next, predicted, self.wm.params['learning_rate'])
        mse = sum((actual_next[i] - predicted[i])**2 for i in range(N_CHANNELS)) / N_CHANNELS
        prediction_reward = 1.0 / (1.0 + mse)

        n_nodes = len(self.wm.nodes)
        metabolic_cost = self.wm.params['metabolic'] * (1 + n_nodes * 0.04)

        # 外部能量（生存用，基于预测）
        self.energy += prediction_reward * 0.5

        action_costs = [0.03, 0.0, 0.01, 0.06]
        action_bonus = {
            0: 0.05 if mse < 0.3 else 0.0,
            1: 0.0, 2: 0.0,
            3: 0.08 if mse < 0.2 else 0.02,
        }
        if 0 <= self._action_chosen < N_ACTIONS:
            metabolic_cost += action_costs[self._action_chosen]
            self.energy += action_bonus.get(self._action_chosen, 0)

        # v9 新增：内在动机奖励（给内在能量，不影响生存但驱动进化选择压）
        intrinsic_r = self.im.update_and_reward(
            self.senses_before_action, actual_next, predicted,
            self._action_chosen if self._action_chosen is not None else 1,
            mse
        )
        # 内在能量本身不直接影响生存，但影响"内在适合度"
        self.intrinsic_log.append(intrinsic_r)

        self.energy -= metabolic_cost
        self.energy = max(-5.0, min(20.0, self.energy))

        self.behavior_trace.append({
            'action': self._action_chosen, 'err': mse, 'energy': self.energy,
            'pred_reward': prediction_reward, 'intrinsic': intrinsic_r,
            'boredom': self.im.boredom, 'drive_energy': self.im.drive_energy,
        })
        if len(self.behavior_trace) > 200: self.behavior_trace = self.behavior_trace[-200:]

        if self.energy <= 0: self.alive = False
        self.age += 1

    def clone_mutate(self, rate=0.15, magnitude=0.25):
        child_g = deepcopy(self.genome)
        n = child_g['params']['n_nodes']
        for i in range(N_CHANNELS):
            for j in range(n):
                if random.random() < rate:
                    child_g['sensory_w'][i][j] += random.uniform(-magnitude, magnitude) * 0.3
                    child_g['sensory_w'][i][j] = max(-5.0, min(5.0, child_g['sensory_w'][i][j]))
        for i in range(n):
            for j in range(N_CHANNELS):
                if random.random() < rate:
                    child_g['predict_w'][i][j] += random.uniform(-magnitude, magnitude) * 0.3
                    child_g['predict_w'][i][j] = max(-5.0, min(5.0, child_g['predict_w'][i][j]))
        for i in range(n):
            for j in range(n):
                if i != j and random.random() < rate * 0.5:
                    child_g['recurrent_w'][i][j] += random.uniform(-magnitude, magnitude) * 0.2
                    child_g['recurrent_w'][i][j] = max(-5.0, min(5.0, child_g['recurrent_w'][i][j]))
        for a in range(N_ACTIONS):
            for j in range(n):
                if random.random() < rate:
                    child_g['action_w'][a][j] += random.uniform(-magnitude, magnitude) * 0.25
                    child_g['action_w'][a][j] = max(-5.0, min(5.0, child_g['action_w'][a][j]))
        if random.random() < rate:
            child_g['params']['wm_decay'] += random.uniform(-0.02, 0.02)
            child_g['params']['wm_decay'] = max(0.5, min(0.98, child_g['params']['wm_decay']))
        if random.random() < rate:
            child_g['params']['learning_rate'] += random.uniform(-0.02, 0.02)
            child_g['params']['learning_rate'] = max(0.001, min(0.5, child_g['params']['learning_rate']))
        if random.random() < rate:
            child_g['params']['metabolic'] += random.uniform(-0.01, 0.01)
            child_g['params']['metabolic'] = max(0.01, min(0.2, child_g['params']['metabolic']))
        if random.random() < rate * 0.8:
            delta = random.choice([-1, 1])
            new_n = max(2, min(MAX_NODES, child_g['params']['n_nodes'] + delta))
            if new_n != child_g['params']['n_nodes']:
                child_g = resize_genome(child_g, new_n)
        child = Organism(genome=child_g, lineage_id=self._gen_id())
        child.parents = [self.lineage_id]
        return child

    def behavior_fingerprint(self):
        if len(self.action_history) < 10: return (0,) * N_ACTIONS + (0,)
        freq = [0] * N_ACTIONS
        for a in self.action_history:
            if 0 <= a < N_ACTIONS: freq[a] += 1
        total = max(1, sum(freq))
        freq_norm = tuple(round(f / total, 3) for f in freq)
        if self.wm.prediction_errors:
            avg_err = sum(self.wm.prediction_errors) / len(self.wm.prediction_errors)
        else: avg_err = 1.0
        return freq_norm + (round(avg_err, 4),)


# ============================================================
# World v9（加内在适合度 + 每代内在统计）
# ============================================================

def genome_distance(g1, g2):
    n1 = g1['params']['n_nodes']; n2 = g2['params']['n_nodes']
    min_n = min(n1, n2)
    dist = 0.0; count = 0
    for i in range(min_n):
        for j in range(N_CHANNELS):
            dist += (g1['predict_w'][i][j] - g2['predict_w'][i][j])**2; count += 1
    for i in range(N_CHANNELS):
        for j in range(min_n):
            dist += (g1['sensory_w'][i][j] - g2['sensory_w'][i][j])**2; count += 1
    for key in ['wm_decay', 'learning_rate', 'metabolic', 'sensory_gain', 'pred_gain']:
        if key in g1['params'] and key in g2['params']:
            dist += (g1['params'][key] - g2['params'][key])**2; count += 1
    dist += ((n1 - n2) * 0.5)**2; count += 1
    return math.sqrt(dist / max(1, count))


class World:
    def __init__(self, n_organisms=30, steps_per_gen=400, difficulty=1.0):
        self.n = n_organisms
        self.steps_per_gen = steps_per_gen
        self.difficulty = difficulty
        self.population = [Organism() for _ in range(n_organisms)]
        self.generation = 0
        self.history = []
        self.lineage_log = []
        self.diversity_log = []
        self.intrinsic_log = []  # v9 新增：记录内在动机演化趋势

    def aggregate_action(self, actions):
        if not actions: return 1
        counts = [0] * N_ACTIONS
        for a in actions:
            if 0 <= a < N_ACTIONS: counts[a] += 1
        return counts.index(max(counts))

    def run_generation(self, mode='training'):
        env = Environment(seed=random.randint(0, 999999), mode=mode, difficulty=self.difficulty)
        for step in range(self.steps_per_gen):
            sensory = env.get_sensory()
            proposals = []
            for org in self.population:
                if not org.alive: continue
                predicted, action = org.plan(sensory)
                proposals.append(action)
            action_code = self.aggregate_action(proposals)
            env.step(action_code)
            actual_next = env.get_sensory()
            for org in self.population:
                if not org.alive: continue
                org.learn_and_update(actual_next, action_code)
        return [o for o in self.population if o.alive]

    def novelty_bonus(self, organism):
        if len(self.population) < 5: return 0.1
        fp = organism.behavior_fingerprint()
        distances = []
        for other in self.population:
            if other is organism or not other.alive: continue
            ofp = other.behavior_fingerprint()
            if len(ofp) != len(fp): continue
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(fp, ofp)))
            distances.append(dist)
        if not distances: return 0.1
        distances.sort()
        k = min(5, len(distances))
        return min(1.0, sum(distances[:k]) / k)

    def compute_fitness(self, organism):
        """v9：内在动机能量参与适合度计算"""
        if not organism.alive and organism.age < 20: return 0.0
        survival = min(1.0, organism.age / 300.0)
        energy_score = min(1.0, max(0, organism.energy) / 10.0) if organism.alive else 0.0
        if organism.wm.prediction_errors:
            avg_err = sum(organism.wm.prediction_errors) / len(organism.wm.prediction_errors)
            prediction_score = 1.0 / (1.0 + avg_err * 5)
        else: prediction_score = 0.0
        novelty = self.novelty_bonus(organism)

        # v9 新增：内在动机适合度 = 洛神"想要"做自己事情的能量
        # 如果它内在有很多好奇/精通/因果感，适合度更高
        intrinsic_fitness = min(1.0, max(0, organism.im.drive_energy + 2.0) / 8.0)

        return (
            0.15 * survival +
            0.10 * energy_score +
            0.35 * prediction_score +
            0.15 * novelty +
            0.25 * intrinsic_fitness  # v9: 内在驱动力占 25% 适合度
        )

    def select_and_reproduce(self):
        fits = [(o, self.compute_fitness(o)) for o in self.population]
        fits.sort(key=lambda x: x[1], reverse=True)
        n_survive = max(4, self.n // 3)
        survivors = [o for o, f in fits[:n_survive]]
        survivor_fits = [f for o, f in fits[:n_survive]]

        new_pop = []
        if survivors and survivor_fits[0] > 0.1:
            elite = survivors[0].clone_mutate(rate=0.02, magnitude=0.03)
            new_pop.append(elite)

        attempts = 0
        while len(new_pop) < self.n and attempts < self.n * 10:
            if survivors and sum(survivor_fits) > 0:
                probs = [max(0.01, f) for f in survivor_fits[:n_survive]]
                sum_p = sum(probs)
                probs = [p / sum_p for p in probs]
                r = random.random(); cum = 0.0
                chosen = survivors[0]
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum: chosen = survivors[i]; break
                new_pop.append(chosen.clone_mutate(rate=0.12, magnitude=0.2))
            else:
                new_pop.append(Organism())
            attempts += 1

        self.population = new_pop[:self.n]
        for o in self.population:
            o.energy = 8.0; o.age = 0; o.alive = True
            o.wm = WorldModel(o.genome); o.im = IntrinsicMotivation()
            o.intrinsic_log = deque(maxlen=50)

    def genome_diversity(self):
        sample = self.population[:15]
        dists = []
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                d = genome_distance(sample[i].genome, sample[j].genome); dists.append(d)
        if not dists: return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        mean_d = sum(dists) / len(dists)
        var_d = sum((d - mean_d)**2 for d in dists) / len(dists)
        return {
            'mean': round(mean_d, 4), 'std': round(math.sqrt(var_d), 4),
            'min': round(min(dists), 4), 'max': round(max(dists), 4),
        }

    def checkpoint(self):
        alive = [o for o in self.population if o.alive]
        if not alive:
            return {
                'gen': self.generation, 'alive': 0, 'alive_ratio': 0.0,
                'best_fitness': 0, 'avg_fitness': 0, 'avg_energy': 0,
                'avg_age': 0, 'best_nodes': 0, 'best_lr': 0,
                'best_metabolic': 0, 'avg_pred_error': 1.0,
                'median_pred_error': 1.0, 'worst_pred_error': 1.0,
                'intrinsic_drive': 0.0, 'avg_boredom': 0.0, 'n_lineages': 0,
            }

        fits = [self.compute_fitness(o) for o in self.population]
        best_i = fits.index(max(fits))
        best = self.population[best_i]
        avg_fit = sum(fits) / len(fits)
        avg_energy = sum(o.energy for o in self.population) / len(self.population)
        avg_age = sum(o.age for o in self.population) / len(self.population)

        # v9：内在统计
        intrinsic_drives = [o.im.drive_energy for o in alive]
        avg_intrinsic = sum(intrinsic_drives) / max(1, len(intrinsic_drives))
        boredoms = [o.im.boredom for o in alive]
        avg_boredom = sum(boredoms) / max(1, len(boredoms))

        pred_errs = []
        for o in alive:
            if o.wm.prediction_errors:
                pred_errs.append(sum(o.wm.prediction_errors) / len(o.wm.prediction_errors))
        pred_errs.sort()
        avg_pred = sum(pred_errs) / max(1, len(pred_errs))
        median_pred = pred_errs[len(pred_errs)//2] if pred_errs else 1.0
        worst_pred = pred_errs[-1] if pred_errs else 1.0

        lin = len(set(o.lineage_id for o in alive))

        return {
            'gen': self.generation, 'alive': len(alive),
            'alive_ratio': len(alive) / self.n,
            'best_fitness': max(fits), 'avg_fitness': avg_fit,
            'avg_energy': avg_energy, 'avg_age': avg_age,
            'best_nodes': best.wm.params['n_nodes'],
            'best_lr': best.wm.params['learning_rate'],
            'best_metabolic': best.wm.params['metabolic'],
            'avg_pred_error': avg_pred,
            'median_pred_error': median_pred,
            'worst_pred_error': worst_pred,
            'intrinsic_drive': round(avg_intrinsic, 4),
            'avg_boredom': round(avg_boredom, 4),
            'n_lineages': lin,
        }

    def run(self, num_gens=200):
        print(f"\n{'='*70}")
        print(f"World-Model Evolution v9 — Intrinsic Motivation / Autonomous Drive")
        print(f"n_orgs={self.n} steps/gen={self.steps_per_gen} difficulty={self.difficulty}")
        print(f"{'='*70}\n")

        for g in range(num_gens):
            self.generation = g
            alive = self.run_generation(mode='training')
            ckpt = self.checkpoint()
            self.history.append(ckpt)
            self.select_and_reproduce()

            if g % 10 == 0:
                div = self.genome_diversity()
                self.diversity_log.append({'gen': g, **div})
            if g % 25 == 0:
                counter = Counter(o.lineage_id for o in alive)
                self.lineage_log.append({'gen': g, 'n_lineages': len(counter), 'shannon': -sum((c/len(alive))*math.log(c/len(alive)) for c in counter.values()) if alive else 0})

            if g % 5 == 0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} | "
                      f"fit={ckpt['best_fitness']:.4f}/{ckpt['avg_fitness']:.4f} | "
                      f"pred={ckpt['avg_pred_error']:.4f}/{ckpt['median_pred_error']:.4f}/{ckpt['worst_pred_error']:.4f} | "
                      f"INTRIN={ckpt['intrinsic_drive']:.3f} borem={ckpt['avg_boredom']:.3f}")

        with open("evolution_log_v9_im.json", "w", encoding="utf-8") as f:
            json.dump({
                'history': self.history, 'lineage_log': self.lineage_log,
                'diversity_log': self.diversity_log, 'num_gens': num_gens,
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n保存到 evolution_log_v9_im.json")
        return self.history[-1]


def run_multi_step_test(world, n_list=None, n_trials=5):
    """复用 v8 协议，测试当前最佳个体的多步前瞻预测"""
    if n_list is None:
        n_list = [1, 3, 5, 10, 20, 50]
    print(f"\n{'='*70}")
    print(f"v9 Multi-step Prediction Test")
    print(f"{'='*70}\n")

    fits = [(o, world.compute_fitness(o)) for o in world.population]
    fits.sort(key=lambda x: x[1], reverse=True)
    best_org = fits[0][0]

    results = {}
    for n_steps in n_list:
        trial_errs = []
        for trial in range(n_trials):
            seed = trial * 777 + 123
            # warmup
            env = Environment(seed=seed, mode='novel')
            best_org.wm.nodes = [0.0] * len(best_org.wm.nodes)
            best_org.wm.last_prediction = [0.0] * N_CHANNELS
            for _ in range(50):
                s = env.get_sensory()
                best_org.wm.forward(s)
                env.step(1)
            # predict n steps
            sensory = env.get_sensory()
            predictions = []
            nodes = list(best_org.wm.nodes)
            curr = sensory[:]
            for n in range(n_steps):
                pred, nodes = best_org.wm.forward_from_nodes(nodes, curr)
                predictions.append(pred[:])
                curr = pred
            # ground truth
            env_gt = Environment(seed=seed, mode='novel')
            for _ in range(50):
                s = env_gt.get_sensory()
                env_gt.step(1)
            gts = []
            for n in range(n_steps):
                env_gt.step(1)
                gts.append(env_gt.get_sensory()[:])
            errs = []
            for n in range(min(n_steps, len(gts))):
                e = sum((predictions[n][i] - gts[n][i])**2 for i in range(N_CHANNELS)) / N_CHANNELS
                errs.append(e)
            if errs:
                trial_errs.append(sum(errs) / len(errs))
        results[str(n_steps)] = sum(trial_errs) / max(1, len(trial_errs)) if trial_errs else 1.0

    for n_str, v in results.items():
        print(f"  N={int(n_str):3d} 步 | EVO-IM={v:.5f}")

    with open("multi_step_prediction_v9.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n保存到 multi_step_prediction_v9.json")
    return results


if __name__ == "__main__":
    random.seed(42)
    world = World(n_organisms=30, steps_per_gen=400, difficulty=1.0)
    final = world.run(num_gens=200)
    run_multi_step_test(world)
