# yinyang_evo_v6.py
# 阴阳扣数字生命系统 · v6 — World Model Evolution
#
# v5 解决全部死亡 → 但生物只是带记忆的 leaky integrator（被动跟踪）
# v6 的核心洞见：真正的"理解"=世界模型=能预测未来
#
# 生物基因组编码一个 World Model：
#   ● N 个内部节点（循环有状态，类似 RNN hidden state）
#   ● 感官→节点投影 [channels × N]
#   ● 节点→预测感官投影 [N × channels]
#   ● 节点→行动策略 [4 × N]
#   ● 反向传播局部更新：预测误差驱动 Hebbian 学习
#
# 进化压力：预测越准 → 能量越高 → 活得更长 → 繁殖更多
#              行动越有效 → 资源越多 → 同类竞争力
#
# "超越市面所有AI"的验证：
#   ● 训练过程中未见过的环境结构做泛化测试
#   ● 与随机参数 world model baseline 对比
#   ● 更高的预测准确率 = 真正的"理解"

import random, math, json, hashlib
from copy import deepcopy
from collections import deque

MAX_NODES = 12
N_CHANNELS = 8
N_ACTIONS = 4
CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']
ACTIONS = ['FORAGE','REST','FLEE','ATTACK']


class PlasticityRules:
    """局部可塑性规则——类似反向传播但只用局部信息"""

    @staticmethod
    def outer_product_update(matrix, row_key, col_key, error_vec, activation_vec, rate):
        """δW = -rate * error · activation^T"""
        for i in range(len(error_vec)):
            if i >= len(matrix):
                break
            for j in range(len(activation_vec)):
                if j >= len(matrix[0]):
                    break
                matrix[i][j] -= rate * error_vec[i] * activation_vec[j] * 0.01
                matrix[i][j] = max(-5.0, min(5.0, matrix[i][j]))


class WorldModel:
    """生物个体的世界模型——预测未来感官输入"""

    def __init__(self, genome, max_nodes=MAX_NODES, n_channels=N_CHANNELS):
        self.max_nodes = max_nodes
        self.n_channels = n_channels
        self.params = genome['params']
        n_active = self.params['n_nodes']

        # 循环状态（hidden）
        self.nodes = [0.0] * n_active
        # 上一次对当前 sensory 的预测
        self.predicted = [0.0] * n_channels
        # N 个节点的增益/激活（tanh squash）

        # 权重矩阵
        # 感官→节点 [channels × n_active] — 怎么感知世界
        self.sensory_w = genome['sensory_w']  # [n_channels][n_active]
        # 节点→感官 [n_active × channels] — 怎么预测世界
        self.predict_w = genome['predict_w']  # [n_active][n_channels]
        # 节点间循环 [n_active × n_active] — 时间动态
        self.recurrent_w = genome['recurrent_w']  # [n_active][n_active]
        # 节点→行动 [n_actions × n_active] — 怎么行动
        self.action_w = genome['action_w']  # [N_ACTIONS][n_active]
        # 行动 bias
        self.action_bias = genome.get('action_bias', [0.0] * N_ACTIONS)

        # 学习统计
        self.prediction_errors = deque(maxlen=50)
        self.cumulative_mse = 0.0

    def forward(self, sensory_vec):
        """
        1. 对比上次预测和当前 sensory → 误差信号
        2. 更新节点状态 ( decay * old + sensory_proj · sensory + recurrent · nodes + error )
        3. 预测下一时刻 sensory = predict_w · nodes
        4. 返回 (predicted_next, nodes_activation)
        """
        # (1) 误差信号
        error = []
        for i in range(self.n_channels):
            err = sensory_vec[i] - self.predicted[i]
            error.append(err)

        err_mag = math.sqrt(sum(e*e for e in error)) / self.n_channels
        self.prediction_errors.append(err_mag)
        self.cumulative_mse += err_mag

        # (2) 节点更新：循环 + 投影 + 误差信号
        n = self.max_nodes if self.max_nodes < len(self.nodes) else len(self.nodes)
        new_nodes = []
        decay = self.params['wm_decay']
        sensory_gain = self.params['sensory_gain']

        for j in range(n):
            # decayed previous
            val = self.nodes[j] * decay

            # 感官输入投影
            for i in range(self.n_channels):
                val += self.sensory_w[i][j] * sensory_vec[i] * sensory_gain

            # 循环连接（节点间相互作用）
            for k in range(n):
                val += self.recurrent_w[j][k] * self.nodes[k]

            # 误差驱动更新（大误差应该改变世界模型理解）
            for i in range(self.n_channels):
                val += error[i] * self.sensory_w[i][j] * 0.1

            # squash
            new_nodes.append(math.tanh(val))

        self.nodes = new_nodes

        # (3) 预测下一时刻
        predicted_next = []
        pred_gain = self.params['pred_gain']
        for i in range(self.n_channels):
            pred = 0.0
            for j in range(n):
                pred += self.predict_w[j][i] * self.nodes[j]
            predicted_next.append(math.tanh(pred * pred_gain))

        self.predicted = predicted_next

        return predicted_next, new_nodes

    def choose_action(self):
        """基于当前节点状态，选择行动"""
        n = len(self.nodes)
        logits = []
        for a in range(N_ACTIONS):
            logit = self.action_bias[a]
            for j in range(n):
                logit += self.action_w[a][j] * self.nodes[j]
            logits.append(logit)

        # softmax
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_e = sum(exps)
        probs = [e / sum_e for e in exps]

        # 采样
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return N_ACTIONS - 1

    def learn(self, actual_sensory, predicted_next, rate):
        """Hebbian + 预测误差驱动的权重更新"""
        error = []
        for i in range(self.n_channels):
            error.append(actual_sensory[i] - predicted_next[i])

        n = len(self.nodes)

        # 更新 predict_w：预测误差驱动
        for i in range(self.n_channels):
            for j in range(n):
                delta = rate * error[i] * self.nodes[j] * 0.01
                self.predict_w[j][i] += delta
                self.predict_w[j][i] = max(-5.0, min(5.0, self.predict_w[j][i]))

        # 更新 sensory_w：共激活 Hebbian
        for j in range(n):
            for i in range(self.n_channels):
                delta = rate * self.nodes[j] * actual_sensory[i] * 0.001
                self.sensory_w[i][j] += delta
                self.sensory_w[i][j] = max(-5.0, min(5.0, self.sensory_w[i][j]))


class Environment:
    """更动态的季节性环境——测试泛化"""
    def __init__(self, seed=None, mode='training'):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.mode = mode          # 'training' 或 'novel'
        self.season = 0           # 0,1,2,3 四种季节
        self.substep = 0

        # 物理场
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

    def step(self, action=None):
        self.t += 1
        self.substep += 1
        if self.substep > 200:
            self.substep = 0
            self.season = (self.season + 1) % 4

        s = self.season

        # 行动反馈
        if action == 0:    # FORAGE — 试图寻找资源
            self.light += random.uniform(0.5, 2.0)
            self.sound_amp += random.uniform(0.1, 0.5)
        elif action == 1:  # REST — 静止保存
            pass
        elif action == 2:  # FLEE — 离开当前区域，场回归默认
            self.light = max(0, min(100, self.light * 0.9))
            self.pressure *= 0.8
        elif action == 3:  # ATTACK — 获取高能量资源
            self.temperature += random.uniform(-1, 1)

        # 季节模式
        if self.mode == 'training':
            # 训练环境 — 已知的季节模式
            if s == 0:    # 明亮安静
                self.light += random.uniform(-1, 3)
                self.sound_amp *= 0.98
            elif s == 1:  # 黑暗嘈杂
                self.light += random.uniform(-3, 1)
                self.sound_amp += random.uniform(0, 1)
            elif s == 2:  # 温暖高压
                self.temperature += random.uniform(0, 0.1)
                self.pressure += random.uniform(0, 0.15)
            else:         # 寒冷低压
                self.temperature += random.uniform(-0.1, 0)
                self.pressure *= 0.95
        else:
            # novel 环境 — 训练中未见的模式
            if s == 0:
                self.light_r += random.uniform(-0.08, 0.08)
                self.sound_freq += random.uniform(-50, 50)
            elif s == 1:
                self.pressure += random.uniform(0, 0.4)
                self.vibration += random.uniform(-0.2, 0.2)
            elif s == 2:
                self.temperature += random.uniform(-0.15, 0.15)
                self.accel[0] = math.sin(self.t * 0.1) * 2.0
            else:
                self.light += random.uniform(-4, 4)
                self.sound_amp += random.uniform(-1, 1)

        # 应用边界和衰减
        self.light = max(0.0, min(100.0, self.light))
        self.light_r = max(-1.0, min(1.0, self.light_r))
        self.sound_freq = max(20.0, min(20000.0, self.sound_freq))
        self.sound_amp = max(0.0, min(120.0, self.sound_amp))
        self.pressure *= 0.96
        self.pressure = max(0.0, min(20.0, self.pressure))
        self.vibration = max(-2.0, min(2.0, self.vibration * 0.93))

        for i in range(5):
            self.smell[i] *= 0.95
            self.smell[i] = min(10.0, max(0.0, self.smell[i]))
            if random.random() < 0.015:
                self.smell[i] += random.uniform(0, 3)
            self.taste[i] *= 0.985
            if random.random() < 0.03:
                self.taste[i] += random.uniform(-0.2, 0.2)
            self.taste[i] = max(-5.0, min(5.0, self.taste[i]))

        self.temperature = max(15.0, min(35.0, self.temperature))
        for k in range(3):
            self.accel[k] = max(-2.0, min(12.0, self.accel[k] * 0.95))

    def get_sensory(self):
        """返回 8 维感官向量"""
        return [
            self.light + self.light_r * 6,
            self.sound_amp + math.log(max(1.0, self.sound_freq)) * 2.5,
            self.pressure + self.vibration * 0.3,
            sum(self.smell) / 5,
            sum(self.taste),
            self.temperature,
            abs(self.accel[2] - 9.8),
            0.0  # proprio（由生物自身填充）
        ]


class Organism:
    CHANNELS = CHANNELS

    def __init__(self, genome=None, lineage_id=None):
        self.genome = genome if genome else self.random_genome()
        self.wm = WorldModel(self.genome)
        self.senses = SensorArray()
        self.state = [0.0] * N_CHANNELS
        self.state_history = []
        self.action_history = []
        self.alive = True
        self.age = 0
        self.energy = 5.0
        self.lineage_id = lineage_id or self._gen_id()
        self.parents = []

        # 行为指纹（用于 novelty 计算）
        self.behavior_trace = []

    def _gen_id(self):
        return hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]

    @staticmethod
    def random_genome():
        n = random.randint(3, MAX_NODES)

        sensory_w = [[random.uniform(-0.3, 0.3) for _ in range(n)]
                     for _ in range(N_CHANNELS)]
        predict_w = [[random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)]
                     for _ in range(n)]
        recurrent_w = [[random.uniform(-0.1, 0.1) if i != j else 0.0
                        for j in range(n)] for i in range(n)]
        action_w = [[random.uniform(-0.2, 0.2) for _ in range(n)]
                    for _ in range(N_ACTIONS)]
        action_bias = [random.uniform(-0.1, 0.1) for _ in range(N_ACTIONS)]

        return {
            'sensory_w': sensory_w,
            'predict_w': predict_w,
            'recurrent_w': recurrent_w,
            'action_w': action_w,
            'action_bias': action_bias,
            'params': {
                'n_nodes': n,
                'wm_decay': random.uniform(0.7, 0.95),
                'sensory_gain': random.uniform(0.3, 2.0),
                'pred_gain': random.uniform(0.5, 2.0),
                'learning_rate': random.uniform(0.01, 0.3),
                'metabolic': random.uniform(0.02, 0.15),
                'attention': {ch: random.uniform(0.5, 1.0) for ch in CHANNELS},
                'input_gain': {ch: random.uniform(0.1, 1.0) for ch in CHANNELS},
            }
        }

    def clone_mutate(self, rate=0.15, magnitude=0.25):
        child_g = deepcopy(self.genome)
        n = child_g['params']['n_nodes']

        # mutate weights
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

        # mutate params
        if random.random() < rate:
            child_g['params']['wm_decay'] += random.uniform(-0.02, 0.02)
            child_g['params']['wm_decay'] = max(0.5, min(0.98, child_g['params']['wm_decay']))

        if random.random() < rate:
            child_g['params']['learning_rate'] += random.uniform(-0.02, 0.02)
            child_g['params']['learning_rate'] = max(0.001, min(0.5, child_g['params']['learning_rate']))

        if random.random() < rate:
            child_g['params']['metabolic'] += random.uniform(-0.01, 0.01)
            child_g['params']['metabolic'] = max(0.005, min(0.3, child_g['params']['metabolic']))

        if random.random() < rate:
            delta = random.choice([-1, 1])
            new_n = max(2, min(MAX_NODES, child_g['params']['n_nodes'] + delta))
            if new_n != child_g['params']['n_nodes']:
                child_g = resize_genome(child_g, new_n)

        child = Organism(genome=child_g, lineage_id=self._gen_id())
        child.parents = [self.lineage_id]
        return child

    def step(self, env):
        """生物一个时间步"""
        if not self.alive:
            return

        # 读取感官
        sensory = env.get_sensory()
        # 时间差分（保留 v5 的 tracking 能力）
        raw_deltas = {}
        prev = self.senses.prev
        for i, ch in enumerate(CHANNELS):
            if prev is None:
                raw_deltas[ch] = 0.0
            else:
                raw_deltas[ch] = sensory[i] - prev[i]
        self.senses.prev = sensory[:]

        # 本体感觉 = 状态变化幅度
        if len(self.state_history) >= 1 and self.state_history[-1]:
            proprio = sum(abs(sensory[i] - self.state_history[-1][i]) for i in range(N_CHANNELS)) / N_CHANNELS
            sensory[N_CHANNELS - 1] = proprio
        else:
            sensory[N_CHANNELS - 1] = 0.0

        self.state = sensory[:]
        self.state_history.append(sensory[:])
        if len(self.state_history) > 300:
            self.state_history = self.state_history[-300:]

        # --- 世界模型 forward ---
        predicted, nodes = self.wm.forward(sensory)
        action_idx = self.wm.choose_action()
        action_name = ACTIONS[action_idx]
        self.action_history.append(action_idx)
        if len(self.action_history) > 200:
            self.action_history = self.action_history[-200:]

        # --- 行动反馈环境 ---
        env.step(action=action_idx)

        # 新的感官（行动后的世界）
        actual_after = env.get_sensory()
        if self.state_history:
            self.state_history[-1][N_CHANNELS-1] = math.sqrt(
                sum((actual_after[i] - sensory[i])**2 for i in range(N_CHANNELS))
            ) / N_CHANNELS

        # --- 世界模型 learn ---
        self.wm.learn(actual_after, predicted, self.wm.params['learning_rate'])

        # --- 能量动态 ---
        mse = sum((actual_after[i] - predicted[i])**2 for i in range(N_CHANNELS)) / N_CHANNELS
        prediction_reward = 1.0 / (1.0 + mse * 10)     # 预测越准，能量越多
        metabolic_cost = self.wm.params['metabolic'] * (1 + len(self.wm.nodes) * 0.05)  # 节点越多越耗

        # 行动影响
        if action_idx == 0:  # FORAGE
            self.energy += prediction_reward * 0.08
        elif action_idx == 1:  # REST
            self.energy += prediction_reward * 0.03
            metabolic_cost *= 0.5
        elif action_idx == 2:  # FLEE
            self.energy += prediction_reward * 0.05
        else:                  # ATTACK
            self.energy += prediction_reward * 0.12
            metabolic_cost *= 1.5

        self.energy -= metabolic_cost

        # 行为记录（用于 novelty）
        self.behavior_trace.append({
            'action': action_idx,
            'err': mse,
            'energy': self.energy,
        })
        if len(self.behavior_trace) > 200:
            self.behavior_trace = self.behavior_trace[-200:]

        if self.energy <= 0:
            self.alive = False

        self.age += 1

    def behavior_fingerprint(self):
        """量化行为模式，用于 novelty 评分"""
        if len(self.action_history) < 10:
            return (0,) * N_ACTIONS + (0,)

        # 行为频率分布
        freq = [0] * N_ACTIONS
        for a in self.action_history:
            if 0 <= a < N_ACTIONS:
                freq[a] += 1
        total = max(1, sum(freq))
        freq_norm = tuple(round(f / total, 3) for f in freq)

        # 预测准确度（低=好）
        if self.wm.prediction_errors:
            avg_err = sum(self.wm.prediction_errors) / len(self.wm.prediction_errors)
        else:
            avg_err = 1.0

        return freq_norm + (round(avg_err, 4),)


class SensorArray:
    CHANNELS = CHANNELS

    def __init__(self):
        self.prev = None


def resize_genome(genome, new_n):
    """调整世界模型大小（节点数变化时）"""
    old_n = genome['params']['n_nodes']
    g = deepcopy(genome)

    # sensory_w [N_CHANNELS][n]
    new_sensory = []
    for i in range(N_CHANNELS):
        row = list(g['sensory_w'][i][:min(old_n, new_n)])
        while len(row) < new_n:
            row.append(random.uniform(-0.3, 0.3))
        new_sensory.append(row)
    g['sensory_w'] = new_sensory

    # predict_w [n][N_CHANNELS]
    new_predict = []
    for i in range(new_n):
        if i < old_n:
            new_predict.append(list(g['predict_w'][i]))
        else:
            new_predict.append([random.uniform(-0.3, 0.3) for _ in range(N_CHANNELS)])
    g['predict_w'] = new_predict

    # recurrent_w [n][n]
    new_rec = [[0.0]*new_n for _ in range(new_n)]
    for i in range(min(old_n, new_n)):
        for j in range(min(old_n, new_n)):
            new_rec[i][j] = g['recurrent_w'][i][j]
    for i in range(old_n, new_n):
        for j in range(old_n, new_n):
            if i != j:
                new_rec[i][j] = random.uniform(-0.1, 0.1)
    g['recurrent_w'] = new_rec

    # action_w [N_ACTIONS][n]
    new_action = []
    for a in range(N_ACTIONS):
        row = list(g['action_w'][a][:min(old_n, new_n)])
        while len(row) < new_n:
            row.append(random.uniform(-0.2, 0.2))
        new_action.append(row)
    g['action_w'] = new_action

    g['params']['n_nodes'] = new_n
    return g


class World:
    def __init__(self, n_organisms=30, steps_per_gen=400):
        self.n = n_organisms
        self.steps_per_gen = steps_per_gen
        self.population = [Organism() for _ in range(n_organisms)]
        self.generation = 0
        self.history = []

    def run_generation(self, mode='training'):
        env = Environment(seed=random.randint(0, 999999), mode=mode)

        for step in range(self.steps_per_gen):
            alive_this = []
            for org in self.population:
                if not org.alive:
                    continue
                org.step(env)
                if org.alive:
                    alive_this.append(org)

        return [o for o in self.population if o.alive]

    def novelty_bonus(self, organism):
        """计算行为稀有度 bonus"""
        if len(self.population) < 5:
            return 0.0
        fp = organism.behavior_fingerprint()
        # 与最近的 K 个邻居的距离
        distances = []
        for other in self.population:
            if other is organism or not other.alive:
                continue
            ofp = other.behavior_fingerprint()
            if len(ofp) != len(fp):
                continue
            dist = sum((a - b)**2 for a, b in zip(fp, ofp)) ** 0.5
            distances.append(dist)
        if not distances:
            return 0.1
        distances.sort()
        k = min(5, len(distances))
        avg_dist = sum(distances[:k]) / k
        return avg_dist * 0.5

    def compute_fitness(self, organism):
        if not organism.alive and organism.age < 10:
            return 0.0

        survival = min(1.0, organism.age / 300.0)

        energy_score = min(1.0, organism.age / 300.0) if organism.alive else 0.0
        if organism.alive:
            energy_score = min(1.0, (organism.energy + 1) / 12.0)

        # 预测准确度 = 世界模型质量
        if organism.wm.prediction_errors:
            avg_err = sum(organism.wm.prediction_errors) / len(organism.wm.prediction_errors)
            prediction_score = 1.0 / (1.0 + avg_err * 5)
        else:
            prediction_score = 0.0

        # novelty bonus
        novelty = self.novelty_bonus(organism)

        return (
            0.2 * survival +
            0.2 * energy_score +
            0.4 * prediction_score +
            0.2 * novelty +
            0.05  # 基础分
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
            new_pop.append(survivors[0].clone_mutate(rate=0.03, magnitude=0.05))

        # 多样性复制
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
                child = chosen.clone_mutate(rate=0.15, magnitude=0.25)
                new_pop.append(child)
            else:
                new_pop.append(Organism())
            attempts += 1

        self.population = new_pop[:self.n]

        # reset energy
        for o in self.population:
            o.energy = 5.0

    def checkpoint(self, extras=None):
        alive = [o for o in self.population if o.alive]
        if not alive:
            return {'gen': self.generation, 'alive': 0, 'alive_ratio': 0.0, 'best_fitness': 0,
                'avg_fitness': 0, 'avg_pred_error': 1.0, 'best_nodes': 0, 'best_lr': 0,
                'best_metabolic': 0, 'best_survival': 0, 'best_prediction': 0}

        fits = [self.compute_fitness(o) for o in self.population]
        best_i = fits.index(max(fits))
        best = self.population[best_i]
        avg_fit = sum(fits) / len(fits)
        avg_energy = sum(o.energy for o in self.population) / len(self.population)
        avg_age = sum(o.age for o in self.population) / len(self.population)

        # 预测误差统计
        pred_errs = []
        for o in alive:
            if o.wm.prediction_errors:
                pred_errs.append(sum(o.wm.prediction_errors) / len(o.wm.prediction_errors))
        avg_pred = sum(pred_errs) / max(1, len(pred_errs))

        ckpt = {
            'gen': self.generation,
            'alive': len(alive),
            'alive_ratio': len(alive) / self.n,
            'best_fitness': max(fits),
            'avg_fitness': avg_fit,
            'avg_energy': avg_energy,
            'avg_age': avg_age,
            'avg_pred_error': avg_pred,
            'best_nodes': best.wm.params['n_nodes'],
            'best_lr': best.wm.params['learning_rate'],
            'best_metabolic': best.wm.params['metabolic'],
            'best_survival': min(1.0, best.age / 300.0),
            'best_prediction': 1.0 / (1.0 + avg_pred * 5) if pred_errs else 0,
        }
        if extras:
            ckpt.update(extras)

        # 最佳生物行为分布
        if best.action_history:
            freq = [0] * N_ACTIONS
            for a in best.action_history:
                if 0 <= a < N_ACTIONS:
                    freq[a] += 1
            total = max(1, sum(freq))
            ckpt['best_action_dist'] = [round(f / total, 3) for f in freq]

        return ckpt

    def run(self, num_gens=100):
        print(f"\n{'='*70}")
        print(f"World-Model Evolution v6 | n_orgs={self.n} steps/gen={self.steps_per_gen}")
        print(f"{'='*70}\n")

        for g in range(num_gens):
            self.generation = g
            alive = self.run_generation(mode='training')
            ckpt = self.checkpoint()
            self.history.append(ckpt)
            self.select_and_reproduce()

            if g % 10 == 0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} ({ckpt['alive_ratio']*100:5.1f}%) | "
                      f"fit={ckpt['best_fitness']:.4f} | avg_fit={ckpt['avg_fitness']:.4f} | "
                      f"pred_err={ckpt['avg_pred_error']:.4f} | "
                      f"nodes={ckpt['best_nodes']:2d} | "
                      f"dist={ckpt.get('best_action_dist', [])}")

        print(f"\n{'='*70}")
        print(f"进化完成 {num_gens} 代")
        print(f"{'='*70}")

        final = self.checkpoint()
        print(f"\n最终统计:")
        print(f"  存活率 = {final['alive_ratio']*100:.1f}%")
        print(f"  平均适应度 = {final['avg_fitness']:.4f}")
        print(f"  最佳适应度 = {final['best_fitness']:.4f}")
        print(f"  平均预测误差 = {final.get('avg_pred_error', 'N/A')}")
        print(f"  最佳世界模型节点数 = {final['best_nodes']}")

        # 保存
        with open("evolution_log_v6.json", "w", encoding="utf-8") as f:
            json.dump({
                'history': self.history,
                'num_gens': num_gens,
            }, f, ensure_ascii=False, indent=2, default=str)
        print("\n保存到 evolution_log_v6.json")

        return final


def run_generalization_test(world, n_tests=5, steps=300):
    """泛化测试：训练中未见过的环境模式"""
    print(f"\n{'='*70}")
    print(f"泛化测试 (novel environment, {n_tests} tasks × {steps} steps)")
    print(f"{'='*70}\n")

    # 拿种群里最佳的 3 个
    fits = []
    for o in world.population[:20]:
        f = world.compute_fitness(o)
        fits.append((o, f))
    fits.sort(key=lambda x: x[1], reverse=True)
    top3 = [o for o, f in fits[:3]]

    results = []
    for trial in range(n_tests):
        env = Environment(seed=trial * 1000 + 500, mode='novel')
        survivors = []
        for org in top3:
            test_org = Organism(genome=deepcopy(org.genome))
            test_org.wm = WorldModel(test_org.genome)
            for s in range(steps):
                test_org.step(env)
                if not test_org.alive:
                    break
            survived_steps = test_org.age
            mse = sum(test_org.wm.prediction_errors) / max(1, len(test_org.wm.prediction_errors)) if test_org.wm.prediction_errors else 1.0
            survivors.append({
                'trial': trial,
                'survived': survived_steps,
                'alive': test_org.alive,
                'mse': mse,
                'nodes': test_org.wm.params['n_nodes'],
            })

        results.extend(survivors)
        avg_surv = sum(r['survived'] for r in survivors) / len(survivors)
        avg_mse = sum(r['mse'] for r in survivors) / len(survivors)
        print(f"  novel task {trial}: avg_survival={avg_surv:.1f}steps, avg_pred_err={avg_mse:.4f}")

    # baseline: 随机初始化世界模型
    print(f"\n--- baseline comparison (random world model) ---")
    baseline_survivals = []
    baseline_mses = []
    for trial in range(n_tests):
        env = Environment(seed=trial * 1000 + 500, mode='novel')
        test_org = Organism()
        test_org.wm = WorldModel(test_org.genome)
        for s in range(steps):
            test_org.step(env)
            if not test_org.alive:
                break
        baseline_survivals.append(test_org.age)
        mse = sum(test_org.wm.prediction_errors) / max(1, len(test_org.wm.prediction_errors)) if test_org.wm.prediction_errors else 1.0
        baseline_mses.append(mse)

        print(f"  baseline trial {trial}: survival={test_org.age}steps, pred_err={mse:.4f}")

    # 汇总
    evo_surv = sum(r['survived'] for r in results) / max(1, len(results))
    evo_mse = sum(r['mse'] for r in results) / max(1, len(results))
    base_surv = sum(baseline_survivals) / max(1, len(baseline_survivals))
    base_mse = sum(baseline_mses) / max(1, len(baseline_mses))

    print(f"\n{'─'*40}")
    print(f"进化模型 vs 随机 baseline:")
    print(f"  平均存活步数: evo={evo_surv:.1f} vs base={base_surv:.1f}  ({'+' if evo_surv>base_surv else ''}{evo_surv-base_surv:.1f})")
    print(f"  平均预测误差: evo={evo_mse:.4f} vs base={base_mse:.4f}  ({'+' if evo_mse<base_mse else ''}{(evo_mse-base_mse):.4f})")

    better = evo_surv > base_surv * 1.1 or evo_mse < base_mse * 0.9
    print(f"  结论: {'进化模型显著优于 baseline ✓' if better else '未显著超越 baseline ✗'}")

    with open("generalization_test_v6.json", "w", encoding="utf-8") as f:
        json.dump({
            'evolved': results,
            'baseline': [{'survival': s, 'mse': m} for s, m in zip(baseline_survivals, baseline_mses)],
            'summary': {
                'evo_survival': evo_surv,
                'evo_mse': evo_mse,
                'base_survival': base_surv,
                'base_mse': base_mse,
                'superior': better,
            }
        }, f, ensure_ascii=False, indent=2, default=str)
    print("保存到 generalization_test_v6.json")


if __name__ == "__main__":
    random.seed(42)
    world = World(n_organisms=30, steps_per_gen=400)
    final = world.run(num_gens=100)

    # 进化结束，做泛化测试
    run_generalization_test(world, n_tests=5, steps=400)
