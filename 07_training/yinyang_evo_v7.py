# yinyang_evo_v7.py
# 阴阳扣数字生命系统 · v7 — World Model Evolution (Fixed)
#
# v6 致命错：每 tick 每个生物各自调 env.step → 30 生物 = 环境 step 30 次/tick
#             行动后读的实际感官 != 预测目标 → 预测永远不准
#
# v7 的同步模型：
#   tick = {
#     1. 环境提供当前感官 S_t（所有生物共享）
#     2. 每个生物：world_model.forward(S_t) → 预测 P_{t+1} + 选择行动 A_t
#     3. 世界：聚合所有 A_t → 环境 step 一次 → S_{t+1}
#     4. 每个生物：对比 P_{t+1} vs S_{t+1} → 学习 + 能量分配
#   }
#
# 选择压力：预测越准 → 能量越多 → 存活/繁殖越多
#           能量代谢 + 节点惩罚 → 逼世界模型的小型化/高效化
#
# "超越市面所有AI"验证：
#   · 泛化到 novel 环境（季节模式训练时未见）
#   · 与随机 baseline 对比预测误差和存活步数
#   · 进化代数 vs 适应度的梯度方向性

import random, math, json, hashlib
from copy import deepcopy
from collections import deque

MAX_NODES = 12
N_CHANNELS = 8
N_ACTIONS = 4
CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']
ACTIONS = ['FORAGE', 'REST', 'FLEE', 'ATTACK']


class WorldModel:
    """生物的世界模型——预测下一时刻感官状态"""

    def __init__(self, genome):
        self.params = genome['params']
        n = self.params['n_nodes']

        # hidden state（循环有状态）
        self.nodes = [0.0] * n
        # 上一次预测（用于 v-error 信号）
        self.last_prediction = [0.0] * N_CHANNELS

        # 权重矩阵
        self.sensory_w = genome['sensory_w']     # [N_CHANNELS][n]
        self.predict_w = genome['predict_w']     # [n][N_CHANNELS]
        self.recurrent_w = genome['recurrent_w'] # [n][n]
        self.action_w = genome['action_w']       # [N_ACTIONS][n]
        self.action_bias = genome['action_bias'] # [N_ACTIONS]

        # 学习记录
        self.prediction_errors = deque(maxlen=80)
        self.cumulative_mse = 0.0
        self.n_predictions = 0

    def forward(self, sensory_vec):
        """
        1. 基于当前 nodes → 预测 next_sensory（不吸收新输入）
        2. 吸收 sensory_vec → 更新 nodes
        返回 predicted_next
        """
        n = len(self.nodes)
        decay = self.params['wm_decay']

        # (1) 预测：当前 nodes → 预测下一感官
        predicted_next = []
        for i in range(N_CHANNELS):
            pred = 0.0
            for j in range(n):
                pred += self.predict_w[j][i] * self.nodes[j]
            predicted_next.append(math.tanh(pred * self.params['pred_gain']))

        # (2) 更新 hidden state（吸收感官 + 循环 + 预测误差）
        new_nodes = []
        for j in range(n):
            val = self.nodes[j] * decay

            # 感官投影
            for i in range(N_CHANNELS):
                val += self.sensory_w[i][j] * sensory_vec[i] * self.params['sensory_gain']

            # 循环连接
            for k in range(n):
                val += self.recurrent_w[j][k] * self.nodes[k]

            # 预测误差驱动（大误差 → 大更新）
            if self.last_prediction and len(self.last_prediction) == N_CHANNELS:
                for i in range(N_CHANNELS):
                    err = sensory_vec[i] - self.last_prediction[i]
                    val += err * self.sensory_w[i][j] * 0.05

            new_nodes.append(math.tanh(val))

        self.nodes = new_nodes
        self.last_prediction = predicted_next[:]
        return predicted_next[:], new_nodes[:]

    def choose_action(self):
        n = len(self.nodes)
        logits = []
        for a in range(N_ACTIONS):
            logit = self.action_bias[a]
            for j in range(n):
                logit += self.action_w[a][j] * self.nodes[j]
            logits.append(logit)

        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_e = sum(exps)
        if sum_e == 0:
            return random.randrange(N_ACTIONS)
        probs = [e / sum_e for e in exps]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return N_ACTIONS - 1

    def forward_from_nodes(self, nodes, sensory_vec):
        """
        给定起始节点状态和感官输入，执行 forward 并返回预测 + 新节点状态。
        用于多步前瞻预测（feeding predictions back）。
        不修改 self.nodes（只读操作）。
        """
        n = len(nodes)
        decay = self.params['wm_decay']

        # 预测当前节点下的下一感官
        predicted_next = []
        for i in range(N_CHANNELS):
            pred = 0.0
            for j in range(min(n, len(self.predict_w))):
                pred += self.predict_w[j][i] * nodes[j]
            predicted_next.append(math.tanh(pred * self.params['pred_gain']))

        # 更新节点（不修改 self.nodes）
        new_nodes = []
        for j in range(n):
            val = nodes[j] * decay
            for i in range(N_CHANNELS):
                if j < len(self.sensory_w[i]):
                    val += self.sensory_w[i][j] * sensory_vec[i] * self.params['sensory_gain']
            for k in range(n):
                if k < len(self.recurrent_w[j]) and j < len(self.recurrent_w):
                    val += self.recurrent_w[j][k] * nodes[k]
            # 预测误差驱动
            if self.last_prediction and len(self.last_prediction) == N_CHANNELS:
                for i in range(N_CHANNELS):
                    if j < len(self.sensory_w[i]):
                        err = sensory_vec[i] - self.last_prediction[i]
                        val += err * self.sensory_w[i][j] * 0.03
            new_nodes.append(math.tanh(val))

        return predicted_next, new_nodes

    def learn(self, actual_next, predicted_next, rate):
        """Hebbian + 误差驱动的 predict_w 更新"""
        n = len(self.nodes)
        errors = [actual_next[i] - predicted_next[i] for i in range(N_CHANNELS)]

        # 记录误差
        mse = sum(e*e for e in errors) / N_CHANNELS
        self.prediction_errors.append(mse)
        self.cumulative_mse += mse
        self.n_predictions += 1

        # 更新 predict_w：W += rate * error ⊗ activation
        for i in range(N_CHANNELS):
            for j in range(n):
                delta = rate * errors[i] * self.nodes[j] * 0.01
                self.predict_w[j][i] += delta
                self.predict_w[j][i] = max(-5.0, min(5.0, self.predict_w[j][i]))

        # 更新 sensory_w — Hebbian
        for j in range(n):
            for i in range(N_CHANNELS):
                delta = rate * self.nodes[j] * actual_next[i] * 0.001
                self.sensory_w[i][j] += delta
                self.sensory_w[i][j] = max(-5.0, min(5.0, self.sensory_w[i][j]))


class Environment:
    def __init__(self, seed=None, mode='training'):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.mode = mode
        self.season = 0
        self.substep = 0

        self.light = 50.0
        self.light_r = 0.0
        self.sound_freq = 440.0
        self.sound_amp = 30.0
        self.pressure = 0.0
        self.vibration = 0.0
        self.smell = [0.0]*5
        self.taste = [0.0]*5
        self.temperature = 20.0
        self.accel = [0.0, 0.0, 9.8]

    def step(self, action_code):
        """环境演化一次——带混沌、尖锐转移、非线性耦合（考验世界模型能力）"""
        self.t += 1
        self.substep += 1
        # 季节每 150 步切换（更频繁，更尖锐）
        if self.substep > 150:
            self.substep = 0
            self.season = (self.season + 1) % 4

        s = self.season

        # ===== 行动影响（反馈到环境） =====
        if action_code == 0:   # FORAGE — 增加可预测资源
            self.light += random.uniform(0.3, 1.0)
            self.sound_amp += random.uniform(0.05, 0.2)
        elif action_code == 2: # FLEE — 消耗能量型逃跑
            self.light *= 0.97
            self.pressure *= 0.92
        elif action_code == 3: # ATTACK — 高风险高回报
            self.temperature += random.uniform(-0.5, 0.5)
            self.pressure += random.uniform(0.05, 0.2)

        # ===== 季节模式（含确定性+随机+混沌成分） =====
        if self.mode == 'training':
            # 确定性正弦（可预测）+ 季节调制 + 混沌 spiking
            base_light = 50 + 25 * math.sin(self.t * 0.02 + s * math.pi / 2)
            noise_light = random.uniform(-2, 2)

            if s == 0:     # 明亮安静季节
                self.light += 0.08 * (base_light - self.light) + noise_light * 0.5
                self.sound_amp += random.uniform(-0.3, 0.1)
                self.pressure += random.uniform(-0.05, 0.05)
                # 周期性脉冲（考验时序记忆）
                if self.substep % 20 == 0:
                    self.sound_amp += random.uniform(3, 6)
            elif s == 1:   # 黑暗嘈杂季节
                self.light += 0.06 * (30 - self.light) + noise_light
                self.sound_amp += random.uniform(-0.2, 0.6)
                # 突发尖峰（需要快速适应）
                if random.random() < 0.02:
                    self.sound_amp += random.uniform(5, 12)
            elif s == 2:   # 温暖低压季节
                self.temperature += 0.06 * (28 - self.temperature) + random.uniform(-0.05, 0.05)
                self.pressure += 0.05 * (3 - self.pressure)
                # 非线性耦合：温度→气味
                temp_effect = (self.temperature - 20) * 0.3
                for i in range(5):
                    self.smell[i] += temp_effect * 0.1
            else:          # 寒冷高压 + 混沌 accel
                self.temperature += 0.06 * (17 - self.temperature)
                self.pressure += 0.05 * (15 - self.pressure) + random.uniform(-0.1, 0.1)
                # 混沌测试：accel 突跳
                if random.random() < 0.08:
                    self.accel[0] += random.uniform(-1.5, 1.5)
                    self.accel[1] += random.uniform(-1.5, 1.5)
        else:
            # Novel 环境：训练中未见过的动态
            if s == 0:     # 双频率混合正弦
                v = 10 * math.sin(self.t * 0.03) + 5 * math.sin(self.t * 0.11)
                self.light += 0.1 * (50 + v - self.light) + random.uniform(-1, 1)
                self.pressure += random.uniform(-0.1, 0.15)
            elif s == 1:   # Logistic 混沌
                x = (self.pressure + 10) / 20   # [0, 1]
                x = 3.7 * x * (1 - x)            # logistic map (chaotic)
                self.pressure = max(0, min(20, x * 20))
                self.sound_amp += random.uniform(-0.5, 0.8)
            elif s == 2:   # 突发 phase shift
                if self.substep == 50:
                    self.light = 100 - self.light     # 突然翻转
                self.temperature += random.uniform(-0.08, 0.08)
                self.sound_freq += random.uniform(-15, 15)
            else:          # 正弦 + 噪声 + 脉冲叠加
                self.light += random.uniform(-3, 3)
                pulse = 3.0 if self.substep % 30 < 3 else 0
                self.sound_amp += random.uniform(-0.8, 0.8) + pulse
                self.accel[2] = 9.8 + 2 * math.sin(self.t * 0.2)

        # 硬边界 + 衰减
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
        """返回 8 维感官向量，已归一化到 [-1, 1] 范围"""
        return [
            (self.light / 50.0) - 1.0,                        # visual: -1 ~ 1
            (self.sound_amp / 60.0) - 0.5 + (math.log(max(1.0, self.sound_freq)) - 6) * 0.15,
            (self.pressure / 10.0) - 0.5,                     # touch
            (sum(self.smell) / 25.0) - 0.5,                   # smell
            (sum(self.taste) / 10.0),                          # taste
            (self.temperature - 25.0) / 10.0,                 # thermal: -1 ~ 1
            (abs(self.accel[2] - 9.8) / 5.0),                 # balance
            0.0                                                # proprio placeholder
        ]


def resize_genome(genome, new_n):
    g = deepcopy(genome)
    old_n = g['params']['n_nodes']

    new_sensory = []
    for i in range(N_CHANNELS):
        row = list(g['sensory_w'][i][:min(old_n, new_n)])
        while len(row) < new_n:
            row.append(random.uniform(-0.3, 0.3))
        new_sensory.append(row)
    g['sensory_w'] = new_sensory

    new_predict = []
    for i in range(new_n):
        if i < old_n:
            new_predict.append(list(g['predict_w'][i]))
        else:
            new_predict.append([random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)])
    g['predict_w'] = new_predict

    new_rec = [[0.0]*new_n for _ in range(new_n)]
    for i in range(min(old_n, new_n)):
        for j in range(min(old_n, new_n)):
            new_rec[i][j] = g['recurrent_w'][i][j]
    for i in range(old_n, new_n):
        for j in range(old_n, new_n):
            if i != j:
                new_rec[i][j] = random.uniform(-0.1, 0.1)
    g['recurrent_w'] = new_rec

    new_action = []
    for a in range(N_ACTIONS):
        row = list(g['action_w'][a][:min(old_n, new_n)])
        while len(row) < new_n:
            row.append(random.uniform(-0.2, 0.2))
        new_action.append(row)
    g['action_w'] = new_action

    g['params']['n_nodes'] = new_n
    return g


class Organism:
    CHANNELS = CHANNELS

    def __init__(self, genome=None, lineage_id=None):
        self.genome = genome if genome else self.random_genome()
        self.wm = WorldModel(self.genome)
        self.senses_prev = None
        self.state = [0.0] * N_CHANNELS
        self.state_history = []
        self.action_history = []
        self.alive = True
        self.age = 0
        self.energy = 8.0
        self.lineage_id = lineage_id or self._gen_id()
        self.parents = []
        self.behavior_trace = []
        self._predicted_next = None
        self._action_chosen = None

    def wm_forward_from_nodes(self, nodes, sensory_vec):
        """Delegate to WorldModel.forward_from_nodes for multi-step prediction."""
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
        """Phase 2: 观察 + 预测 + 选择行动"""
        # 本体感觉
        if self.senses_prev is not None:
            proprio = math.sqrt(sum((sensory[i] - self.senses_prev[i])**2 for i in range(N_CHANNELS))) / N_CHANNELS
            sensory[N_CHANNELS - 1] = proprio
        else:
            sensory[N_CHANNELS - 1] = 0.0

        self.senses_prev = sensory[:]
        self.state = sensory[:]
        self.state_history.append(sensory[:])
        if len(self.state_history) > 300:
            self.state_history = self.state_history[-300:]

        # 世界模型 forward
        predicted, nodes = self.wm.forward(sensory)
        self._predicted_next = predicted

        # 选择行动
        action_idx = self.wm.choose_action()
        self._action_chosen = action_idx
        self.action_history.append(action_idx)
        if len(self.action_history) > 200:
            self.action_history = self.action_history[-200:]

        return predicted, action_idx

    def learn_and_update(self, actual_next, action_code):
        """Phase 4: 对比实际 → 学习 + 能量动态"""
        if not self.alive:
            return

        predicted = self._predicted_next
        if predicted is None:
            return

        # 学习
        self.wm.learn(actual_next, predicted, self.wm.params['learning_rate'])

        # 预测误差（已归一化感官，mse 通常在 0-4 范围）
        mse = sum((actual_next[i] - predicted[i])**2 for i in range(N_CHANNELS)) / N_CHANNELS
        prediction_reward = 1.0 / (1.0 + mse)          # [0.2, 1.0]

        # 代谢成本
        n_nodes = len(self.wm.nodes)
        metabolic_cost = self.wm.params['metabolic'] * (1 + n_nodes * 0.04)

        # 基础能量输入（世界模型通过预测获得能量——理解世界=生存）
        # 预测准确度越高 → 获得能量越多
        self.energy += prediction_reward * 0.5          # 世界模型驱动的能源

        # 行动成本
        action_costs = [0.03, 0.0, 0.01, 0.06]
        action_bonus = {
            0: 0.05 if mse < 0.3 else 0.0,     # FORAGE bonus when prediction good
            1: 0.0,                              # REST cheap
            2: 0.0,                              # FLEE cheap
            3: 0.08 if mse < 0.2 else 0.02,     # ATTACK bonus when prediction excellent
        }
        if 0 <= self._action_chosen < N_ACTIONS:
            metabolic_cost += action_costs[self._action_chosen]
            self.energy += action_bonus.get(self._action_chosen, 0)

        self.energy -= metabolic_cost
        self.energy = max(-5.0, min(20.0, self.energy))   # 能量有界

        self.behavior_trace.append({
            'action': self._action_chosen,
            'err': mse,
            'energy': self.energy,
            'pred_reward': prediction_reward,
        })
        if len(self.behavior_trace) > 200:
            self.behavior_trace = self.behavior_trace[-200:]

        if self.energy <= 0:
            self.alive = False

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
        if len(self.action_history) < 10:
            return (0,) * N_ACTIONS + (0,)
        freq = [0] * N_ACTIONS
        for a in self.action_history:
            if 0 <= a < N_ACTIONS:
                freq[a] += 1
        total = max(1, sum(freq))
        freq_norm = tuple(round(f / total, 3) for f in freq)
        if self.wm.prediction_errors:
            avg_err = sum(self.wm.prediction_errors) / len(self.wm.prediction_errors)
        else:
            avg_err = 1.0
        return freq_norm + (round(avg_err, 4),)


class World:
    def __init__(self, n_organisms=30, steps_per_gen=400):
        self.n = n_organisms
        self.steps_per_gen = steps_per_gen
        self.population = [Organism() for _ in range(n_organisms)]
        self.generation = 0
        self.history = []

    def aggregate_action(self, actions):
        """聚合所有生物的行动 → 一个代表行动"""
        if not actions:
            return 1
        counts = [0] * N_ACTIONS
        for a in actions:
            if 0 <= a < N_ACTIONS:
                counts[a] += 1
        return counts.index(max(counts))

    def run_generation(self, mode='training'):
        env = Environment(seed=random.randint(0, 999999), mode=mode)

        for step in range(self.steps_per_gen):
            sensory = env.get_sensory()

            # Phase 2: 每个生物 plan（观察 + 预测 + 选行动）
            proposals = []
            for org in self.population:
                if not org.alive:
                    continue
                predicted, action = org.plan(sensory)
                proposals.append(action)

            # Phase 3: 环境 step 一次
            action_code = self.aggregate_action(proposals)
            env.step(action_code)

            # Phase 4: 所有生物读实际 + 学习
            actual_next = env.get_sensory()
            for org in self.population:
                if not org.alive:
                    continue
                org.learn_and_update(actual_next, action_code)

        return [o for o in self.population if o.alive]

    def novelty_bonus(self, organism):
        if len(self.population) < 5:
            return 0.1
        fp = organism.behavior_fingerprint()
        distances = []
        for other in self.population:
            if other is organism or not other.alive:
                continue
            ofp = other.behavior_fingerprint()
            if len(ofp) != len(fp):
                continue
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(fp, ofp)))
            distances.append(dist)
        if not distances:
            return 0.1
        distances.sort()
        k = min(5, len(distances))
        return min(1.0, sum(distances[:k]) / k)

    def compute_fitness(self, organism):
        if not organism.alive and organism.age < 20:
            return 0.0
        survival = min(1.0, organism.age / 300.0)
        energy_score = min(1.0, max(0, organism.energy) / 10.0) if organism.alive else 0.0

        # 预测准确度 = 世界模型质量
        if organism.wm.prediction_errors:
            avg_err = sum(organism.wm.prediction_errors) / len(organism.wm.prediction_errors)
            prediction_score = 1.0 / (1.0 + avg_err * 5)
        else:
            prediction_score = 0.0

        novelty = self.novelty_bonus(organism)

        return (
            0.20 * survival +
            0.15 * energy_score +
            0.45 * prediction_score +
            0.20 * novelty
        )

    def select_and_reproduce(self):
        fits = [(o, self.compute_fitness(o)) for o in self.population]
        fits.sort(key=lambda x: x[1], reverse=True)

        n_survive = max(4, self.n // 3)
        survivors = [o for o, f in fits[:n_survive]]
        survivor_fits = [f for o, f in fits[:n_survive]]

        new_pop = []

        # elitism
        if survivors and survivor_fits[0] > 0.1:
            elite = survivors[0].clone_mutate(rate=0.02, magnitude=0.03)
            new_pop.append(elite)

        attempts = 0
        while len(new_pop) < self.n and attempts < self.n * 10:
            if survivors and sum(survivor_fits) > 0:
                probs = [max(0.01, f) for f in survivor_fits[:n_survive]]
                sum_p = sum(probs)
                probs = [p / sum_p for p in probs]
                r = random.random()
                cum = 0.0
                chosen = survivors[0]
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        chosen = survivors[i]
                        break
                child = chosen.clone_mutate(rate=0.12, magnitude=0.2)
                new_pop.append(child)
            else:
                new_pop.append(Organism())
            attempts += 1

        self.population = new_pop[:self.n]
        for o in self.population:
            o.energy = 8.0
            o.age = 0
            o.alive = True
            o.wm = WorldModel(o.genome)

    def checkpoint(self):
        alive = [o for o in self.population if o.alive]
        if not alive:
            return {'gen': self.generation, 'alive': 0, 'alive_ratio': 0.0,
                    'best_fitness': 0, 'avg_fitness': 0, 'avg_energy': 0,
                    'avg_age': 0, 'best_nodes': 0, 'best_lr': 0,
                    'best_metabolic': 0, 'avg_pred_error': 1.0}

        fits = [self.compute_fitness(o) for o in self.population]
        best_i = fits.index(max(fits))
        best = self.population[best_i]
        avg_fit = sum(fits) / len(fits)
        avg_energy = sum(o.energy for o in self.population) / len(self.population)
        avg_age = sum(o.age for o in self.population) / len(self.population)

        pred_errs = []
        for o in alive:
            if o.wm.prediction_errors:
                pred_errs.append(sum(o.wm.prediction_errors) / len(o.wm.prediction_errors))
        avg_pred = sum(pred_errs) / max(1, len(pred_errs))

        return {
            'gen': self.generation,
            'alive': len(alive),
            'alive_ratio': len(alive) / self.n,
            'best_fitness': max(fits),
            'avg_fitness': avg_fit,
            'avg_energy': avg_energy,
            'avg_age': avg_age,
            'best_nodes': best.wm.params['n_nodes'],
            'best_lr': best.wm.params['learning_rate'],
            'best_metabolic': best.wm.params['metabolic'],
            'avg_pred_error': avg_pred,
        }

    def run(self, num_gens=100):
        print(f"\n{'='*70}")
        print(f"World-Model Evolution v7 | n_orgs={self.n} steps/gen={self.steps_per_gen}")
        print(f"{'='*70}\n")

        for g in range(num_gens):
            self.generation = g
            alive = self.run_generation(mode='training')
            ckpt = self.checkpoint()
            self.history.append(ckpt)
            self.select_and_reproduce()

            if g % 5 == 0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} ({ckpt['alive_ratio']*100:5.1f}%) | "
                      f"fit={ckpt['best_fitness']:.4f} | avg_fit={ckpt['avg_fitness']:.4f} | "
                      f"pred_err={ckpt['avg_pred_error']:.4f} | nodes={ckpt['best_nodes']:2d} | "
                      f"lr={ckpt['best_lr']:.4f} | metab={ckpt['best_metabolic']:.4f}")

        print(f"\n{'='*70}")
        print(f"进化完成 {num_gens} 代")
        print(f"{'='*70}")

        # history[-1] = 最后一个 gen 的真实数据（select/reproduce 之前）
        final = self.history[-1]
        print(f"\n最终统计 (gen {final['gen']}):")
        print(f"  存活率 = {final['alive_ratio']*100:.1f}%")
        print(f"  平均适应度 = {final['avg_fitness']:.4f}")
        print(f"  最佳适应度 = {final['best_fitness']:.4f}")
        print(f"  平均预测误差 = {final.get('avg_pred_error', 'N/A')}")

        with open("evolution_log_v7.json", "w", encoding="utf-8") as f:
            json.dump({'history': self.history, 'num_gens': num_gens}, f,
                      ensure_ascii=False, indent=2, default=str)
        print("保存到 evolution_log_v7.json")

        return final


def run_generalization_test(world, n_tests=5, steps=400):
    """泛化测试：novel 环境 + baseline 对比"""
    print(f"\n{'='*70}")
    print(f"泛化测试 & baseline 对比")
    print(f"{'='*70}\n")

    fits = [(o, world.compute_fitness(o)) for o in world.population[:20]]
    fits.sort(key=lambda x: x[1], reverse=True)
    top3 = [o for o, f in fits[:3]]

    evo_results = []
    for trial in range(n_tests):
        env = Environment(seed=trial * 1000 + 500, mode='novel')
        for org in top3:
            test_org = Organism(genome=deepcopy(org.genome))
            test_org.wm = WorldModel(test_org.genome)
            for s in range(steps):
                sensory = env.get_sensory()
                predicted, action = test_org.plan(sensory)
                env.step(action)
                actual_next = env.get_sensory()
                test_org.learn_and_update(actual_next, action)
                if not test_org.alive:
                    break

            avg_mse = sum(test_org.wm.prediction_errors) / max(1, len(test_org.wm.prediction_errors)) if test_org.wm.prediction_errors else 1.0
            evo_results.append({'trial': trial, 'survived': test_org.age, 'alive': test_org.alive, 'mse': avg_mse})
            print(f"  evo trial {trial}: survival={test_org.age}, pred_err={avg_mse:.4f}, alive={test_org.alive}")

    baseline_results = []
    for trial in range(n_tests):
        env = Environment(seed=trial * 1000 + 500, mode='novel')
        test_org = Organism()
        for s in range(steps):
            sensory = env.get_sensory()
            predicted, action = test_org.plan(sensory)
            env.step(action)
            actual_next = env.get_sensory()
            test_org.learn_and_update(actual_next, action)
            if not test_org.alive:
                break
        avg_mse = sum(test_org.wm.prediction_errors) / max(1, len(test_org.wm.prediction_errors)) if test_org.wm.prediction_errors else 1.0
        baseline_results.append({'trial': trial, 'survived': test_org.age, 'alive': test_org.alive, 'mse': avg_mse})
        print(f"  baseline trial {trial}: survival={test_org.age}, pred_err={avg_mse:.4f}, alive={test_org.alive}")

    evo_surv = sum(r['survived'] for r in evo_results) / max(1, len(evo_results))
    evo_mse = sum(r['mse'] for r in evo_results) / max(1, len(evo_results))
    base_surv = sum(r['survived'] for r in baseline_results) / max(1, len(baseline_results))
    base_mse = sum(r['mse'] for r in baseline_results) / max(1, len(baseline_results))

    print(f"\n{'─'*40}")
    print(f"进化模型 vs 随机 baseline:")
    print(f"  存活步数:  evo={evo_surv:.1f}  base={base_surv:.1f}  (Δ{evo_surv-base_surv:+.1f})")
    print(f"  预测误差:  evo={evo_mse:.4f}  base={base_mse:.4f}  (Δ{evo_mse-base_mse:+.4f})")

    surv_better = evo_surv > base_surv * 1.1
    pred_better = evo_mse < base_mse * 0.9
    print(f"  存活优势:  {'✓ 进化模型显著优于 baseline' if surv_better else '✗ 未显著优于 baseline'}")
    print(f"  预测优势:  {'✓ 进化模型预测更准' if pred_better else '✗ 预测未优于 baseline'}")

    with open("generalization_test_v7.json", "w", encoding="utf-8") as f:
        json.dump({
            'evolved': evo_results,
            'baseline': baseline_results,
            'summary': {
                'evo_survival': evo_surv, 'evo_mse': evo_mse,
                'base_survival': base_surv, 'base_mse': base_mse,
                'survival_better': surv_better, 'prediction_better': pred_better,
            }
        }, f, ensure_ascii=False, indent=2, default=str)
    print("保存到 generalization_test_v7.json")


def run_multi_step_prediction_test(world, n_steps_list=[1, 3, 5, 10, 20], n_trials=5):
    """
    多步前瞻预测测试——检验世界模型的真实"理解"深度。
    好的世界模型应该能准确预测 N 步后的状态 (feeding own predictions back)。
    baseline（随机模型）的误差会随 N 指数增长，进化模型应更持久。
    """
    print(f"\n{'='*70}")
    print(f"多步前瞻预测测试")
    print(f"{'='*70}\n")

    fits = [(o, world.compute_fitness(o)) for o in world.population[:20]]
    fits.sort(key=lambda x: x[1], reverse=True)
    best_org = fits[0][0]

    results = {'evolved': {}, 'baseline': {}}

    for n_steps in n_steps_list:
        evo_errs = []
        base_errs = []

        for trial in range(n_trials):
            seed = trial * 777 + 123

            # --- 进化模型 ---
            env = Environment(seed=seed, mode='novel')
            test_org = Organism(genome=deepcopy(best_org.genome))
            # 先跑 50 步让 hidden state 稳定
            for _ in range(50):
                sensory = env.get_sensory()
                test_org.plan(sensory)
                env.step(1)  # REST
                actual = env.get_sensory()
                test_org.learn_and_update(actual, 1)

            # 多步前瞻
            sensory = env.get_sensory()
            predictions = []
            state_nodes = list(test_org.wm.nodes)
            current_pred = sensory[:]
            for n in range(n_steps):
                pred, state_nodes = test_org.wm_forward_from_nodes(state_nodes, current_pred)
                predictions.append(pred[:])
                current_pred = pred

            # 实际跑 n_steps 步
            env_multi = Environment(seed=seed, mode='novel')
            for _ in range(50):
                s = env_multi.get_sensory()
                env_multi.step(1)
            actuals = [env_multi.get_sensory()[:]]
            for n in range(n_steps):
                env_multi.step(1)
                actuals.append(env_multi.get_sensory()[:])

            # 计算 N 步前瞻误差
            errs = []
            for n in range(min(n_steps, len(actuals) - 1)):
                e = sum((predictions[n][i] - actuals[n + 1][i]) ** 2 for i in range(N_CHANNELS)) / N_CHANNELS
                errs.append(e)
            if errs:
                evo_errs.append(sum(errs) / len(errs))

            # --- Baseline (random) ---
            env = Environment(seed=seed, mode='novel')
            base_org = Organism()
            for _ in range(50):
                sensory = env.get_sensory()
                base_org.plan(sensory)
                env.step(1)
                actual = env.get_sensory()
                base_org.learn_and_update(actual, 1)

            sensory = env.get_sensory()
            predictions = []
            state_nodes = list(base_org.wm.nodes)
            current_pred = sensory[:]
            for n in range(n_steps):
                pred, state_nodes = base_org.wm_forward_from_nodes(state_nodes, current_pred)
                predictions.append(pred[:])
                current_pred = pred

            env_multi = Environment(seed=seed, mode='novel')
            for _ in range(50):
                s = env_multi.get_sensory()
                env_multi.step(1)
            actuals = [env_multi.get_sensory()[:]]
            for n in range(n_steps):
                env_multi.step(1)
                actuals.append(env_multi.get_sensory()[:])

            errs = []
            for n in range(min(n_steps, len(actuals) - 1)):
                e = sum((predictions[n][i] - actuals[n + 1][i]) ** 2 for i in range(N_CHANNELS)) / N_CHANNELS
                errs.append(e)
            if errs:
                base_errs.append(sum(errs) / len(errs))

        evo_avg = sum(evo_errs) / max(1, len(evo_errs))
        base_avg = sum(base_errs) / max(1, len(base_errs))
        results['evolved'][n_steps] = evo_avg
        results['baseline'][n_steps] = base_avg
        ratio = base_avg / max(1e-8, evo_avg) if evo_avg > 0 else float('inf')
        print(f"  N={n_steps:3d} 步 | 进化={evo_avg:.5f}  baseline={base_avg:.5f}  (baseline/进化={ratio:.1f}x)")

    with open("multi_step_prediction_v7.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("保存到 multi_step_prediction_v7.json")
    return results


if __name__ == "__main__":
    random.seed(42)
    world = World(n_organisms=30, steps_per_gen=400)
    final = world.run(num_gens=100)
    run_generalization_test(world, n_tests=5, steps=400)
    run_multi_step_prediction_test(world)
