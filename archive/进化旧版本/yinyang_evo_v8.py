# yinyang_evo_v8.py
# 阴阳扣数字生命系统 · v8 — 200 代进化 + 瓶颈检测
#
# 从 v7 升级：
#   · 目标 200 代（v7 只做 30-100）
#   · 每 25 代保存完整 world state (支持断点续跑)
#   · 谱系多样性追踪（lineage_id 计数 / Shannon 熵）
#   · 群体基因组距离（每 10 代算一次 pairwise 距离矩阵）
#   · 更细粒度的预测误差分布（best / median / worst）
#   · 测试环境压力升级方案（检测到平台化后自动增加难度）

import random, math, json, hashlib, os, pickle
from copy import deepcopy
from collections import deque, Counter

MAX_NODES = 12
N_CHANNELS = 8
N_ACTIONS = 4
CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']
ACTIONS = ['FORAGE', 'REST', 'FLEE', 'ATTACK']

CHECKPOINT_FILE = "checkpoint_v8.pkl"


class WorldModel:
    """生物的世界模型——预测下一时刻感官状态"""

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

        # (1) 预测
        predicted_next = []
        for i in range(N_CHANNELS):
            pred = 0.0
            for j in range(n):
                pred += self.predict_w[j][i] * self.nodes[j]
            predicted_next.append(math.tanh(pred * self.params['pred_gain']))

        # (2) 更新 hidden state
        new_nodes = []
        for j in range(n):
            val = self.nodes[j] * decay
            for i in range(N_CHANNELS):
                val += self.sensory_w[i][j] * sensory_vec[i] * self.params['sensory_gain']
            for k in range(n):
                val += self.recurrent_w[j][k] * self.nodes[k]
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
        n = len(nodes)
        decay = self.params['wm_decay']

        predicted_next = []
        for i in range(N_CHANNELS):
            pred = 0.0
            for j in range(min(n, len(self.predict_w))):
                pred += self.predict_w[j][i] * nodes[j]
            predicted_next.append(math.tanh(pred * self.params['pred_gain']))

        new_nodes = []
        for j in range(n):
            val = nodes[j] * decay
            for i in range(N_CHANNELS):
                if j < len(self.sensory_w[i]):
                    val += self.sensory_w[i][j] * sensory_vec[i] * self.params['sensory_gain']
            for k in range(n):
                if k < len(self.recurrent_w[j]) and j < len(self.recurrent_w):
                    val += self.recurrent_w[j][k] * nodes[k]
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


class Environment:
    def __init__(self, seed=None, mode='training', difficulty=1.0):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.mode = mode
        self.difficulty = difficulty
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
        self.t += 1
        self.substep += 1
        season_period = max(50, int(150 / self.difficulty))
        if self.substep > season_period:
            self.substep = 0
            self.season = (self.season + 1) % 4

        s = self.season

        # 行动影响
        if action_code == 0:
            self.light += random.uniform(0.3, 1.0)
            self.sound_amp += random.uniform(0.05, 0.2)
        elif action_code == 2:
            self.light *= 0.97
            self.pressure *= 0.92
        elif action_code == 3:
            self.temperature += random.uniform(-0.5, 0.5)
            self.pressure += random.uniform(0.05, 0.2)

        diff = self.difficulty

        if self.mode == 'training':
            base_light = 50 + 25 * math.sin(self.t * 0.02 + s * math.pi / 2)
            noise_mult = 0.5 + diff * 0.5
            noise_light = random.uniform(-2, 2) * noise_mult

            if s == 0:
                self.light += 0.08 * (base_light - self.light) + noise_light
                self.sound_amp += random.uniform(-0.3, 0.1) * diff
                self.pressure += random.uniform(-0.05, 0.05) * diff
                if self.substep % max(5, int(20 / diff)) == 0:
                    self.sound_amp += random.uniform(3, 6) * diff
            elif s == 1:
                self.light += 0.06 * (30 - self.light) + noise_light
                self.sound_amp += random.uniform(-0.2, 0.6) * diff
                spike_prob = 0.02 * diff
                if random.random() < spike_prob:
                    self.sound_amp += random.uniform(5, 12) * diff
            elif s == 2:
                self.temperature += 0.06 * (28 - self.temperature) + random.uniform(-0.05, 0.05) * diff
                self.pressure += 0.05 * (3 - self.pressure)
                temp_effect = (self.temperature - 20) * 0.3
                for i in range(5):
                    self.smell[i] += temp_effect * 0.1 * diff
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
                if self.substep == 50:
                    self.light = 100 - self.light
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

        self.senses_prev = sensory[:]
        self.state = sensory[:]
        self.state_history.append(sensory[:])
        if len(self.state_history) > 300:
            self.state_history = self.state_history[-300:]

        predicted, nodes = self.wm.forward(sensory)
        self._predicted_next = predicted
        action_idx = self.wm.choose_action()
        self._action_chosen = action_idx
        self.action_history.append(action_idx)
        if len(self.action_history) > 200:
            self.action_history = self.action_history[-200:]

        return predicted, action_idx

    def learn_and_update(self, actual_next, action_code):
        if not self.alive:
            return
        predicted = self._predicted_next
        if predicted is None:
            return

        self.wm.learn(actual_next, predicted, self.wm.params['learning_rate'])
        mse = sum((actual_next[i] - predicted[i])**2 for i in range(N_CHANNELS)) / N_CHANNELS
        prediction_reward = 1.0 / (1.0 + mse)

        n_nodes = len(self.wm.nodes)
        metabolic_cost = self.wm.params['metabolic'] * (1 + n_nodes * 0.04)
        self.energy += prediction_reward * 0.5

        action_costs = [0.03, 0.0, 0.01, 0.06]
        action_bonus = {
            0: 0.05 if mse < 0.3 else 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.08 if mse < 0.2 else 0.02,
        }
        if 0 <= self._action_chosen < N_ACTIONS:
            metabolic_cost += action_costs[self._action_chosen]
            self.energy += action_bonus.get(self._action_chosen, 0)

        self.energy -= metabolic_cost
        self.energy = max(-5.0, min(20.0, self.energy))

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


def genome_distance(g1, g2):
    """两个基因组之间的数值距离（用于多样性检测）"""
    n1 = g1['params']['n_nodes']
    n2 = g2['params']['n_nodes']
    min_n = min(n1, n2)

    dist = 0.0
    count = 0

    # predict_w 重叠部分
    for i in range(min_n):
        for j in range(N_CHANNELS):
            dist += (g1['predict_w'][i][j] - g2['predict_w'][i][j]) ** 2
            count += 1

    # sensory_w 重叠部分
    for i in range(N_CHANNELS):
        for j in range(min_n):
            dist += (g1['sensory_w'][i][j] - g2['sensory_w'][i][j]) ** 2
            count += 1

    # param 距离
    for key in ['wm_decay', 'learning_rate', 'metabolic', 'sensory_gain', 'pred_gain']:
        if key in g1['params'] and key in g2['params']:
            dist += (g1['params'][key] - g2['params'][key]) ** 2
            count += 1

    # 节点数差异
    dist += ((n1 - n2) * 0.5) ** 2
    count += 1

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

    def aggregate_action(self, actions):
        if not actions:
            return 1
        counts = [0] * N_ACTIONS
        for a in actions:
            if 0 <= a < N_ACTIONS:
                counts[a] += 1
        return counts.index(max(counts))

    def run_generation(self, mode='training'):
        env = Environment(seed=random.randint(0, 999999), mode=mode, difficulty=self.difficulty)

        for step in range(self.steps_per_gen):
            sensory = env.get_sensory()
            proposals = []
            for org in self.population:
                if not org.alive:
                    continue
                predicted, action = org.plan(sensory)
                proposals.append(action)

            action_code = self.aggregate_action(proposals)
            env.step(action_code)
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

    def lineage_stats(self):
        """统计群体中各谱系的占比"""
        alive = [o for o in self.population if o.alive]
        if not alive:
            return {'n_lineages': 0, 'shannon': 0.0, 'dominant_pct': 0.0}

        counter = Counter(o.lineage_id for o in alive)
        total = len(alive)
        counts = list(counter.values())
        n_lineages = len(counter)

        # Shannon 熵（归一化到 [0,1]）
        probs = [c / total for c in counts]
        shannon = -sum(p * math.log(max(1e-10, p)) for p in probs)
        shannon_norm = shannon / math.log(max(2, n_lineages)) if n_lineages > 1 else 0.0

        dominant_pct = max(counts) / total if counts else 0.0

        return {
            'n_lineages': n_lineages,
            'shannon': round(shannon_norm, 4),
            'dominant_pct': round(dominant_pct, 4),
            'lineage_counts': dict(counter.most_common(5)),
        }

    def genome_diversity(self):
        """群体基因组 pairwise 距离统计"""
        sample = self.population[:15]
        dists = []
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                d = genome_distance(sample[i].genome, sample[j].genome)
                dists.append(d)
        if not dists:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        mean_d = sum(dists) / len(dists)
        var_d = sum((d - mean_d)**2 for d in dists) / len(dists)
        return {
            'mean': round(mean_d, 4),
            'std': round(math.sqrt(var_d), 4),
            'min': round(min(dists), 4),
            'max': round(max(dists), 4),
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
                'n_lineages': 0, 'shannon': 0.0,
            }

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

        pred_errs.sort()
        avg_pred = sum(pred_errs) / max(1, len(pred_errs))
        median_pred = pred_errs[len(pred_errs)//2] if pred_errs else 1.0
        worst_pred = pred_errs[-1] if pred_errs else 1.0

        lin = self.lineage_stats()

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
            'median_pred_error': median_pred,
            'worst_pred_error': worst_pred,
            'n_lineages': lin['n_lineages'],
            'shannon': lin['shannon'],
            'dominant_pct': lin['dominant_pct'],
        }

    def save_checkpoint(self):
        """保存完整断点"""
        state = {
            'generation': self.generation,
            'population': self.population,
            'history': self.history,
            'difficulty': self.difficulty,
            'rng_state': random.getstate(),
        }
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(state, f)

    def detect_plateau(self, window=10):
        """检测适应度平台化"""
        if len(self.history) < window + 1:
            return False
        recent = self.history[-window:]
        fits = [h['avg_fitness'] for h in recent]
        fit_range = max(fits) - min(fits)
        return fit_range < 0.005  # 10 代内 fitness 变化 < 0.5% = 平台化

    def run(self, num_gens=200, resume=True):
        print(f"\n{'='*70}")
        print(f"World-Model Evolution v8 | n_orgs={self.n} steps/gen={self.steps_per_gen}")
        print(f"目标: {num_gens} 代 | 难度: {self.difficulty}")
        print(f"{'='*70}\n")

        # Try resume
        start_gen = 0
        if resume and os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'rb') as f:
                    state = pickle.load(f)
                self.generation = state['generation']
                self.population = state['population']
                self.history = state['history']
                self.difficulty = state.get('difficulty', 1.0)
                random.setstate(state.get('rng_state', random.getstate()))
                start_gen = self.generation + 1
                print(f"断点续跑：从第 {start_gen} 代继续")
            except Exception as e:
                print(f"断点加载失败 ({e})，从头开始")

        for g in range(start_gen, num_gens):
            self.generation = g
            alive = self.run_generation(mode='training')
            ckpt = self.checkpoint()
            self.history.append(ckpt)
            self.select_and_reproduce()

            # 每 10 代计算基因组多样性
            if g % 10 == 0:
                div = self.genome_diversity()
                self.diversity_log.append({'gen': g, **div})

            # 谱系追踪
            if g % 25 == 0:
                lin = self.lineage_stats()
                self.lineage_log.append({'gen': g, **lin})
                # 保存断点
                self.save_checkpoint()

            # 平台化检测 → 增加难度
            if g > 0 and g % 50 == 0 and self.detect_plateau(window=15):
                old_diff = self.difficulty
                self.difficulty = min(3.0, self.difficulty * 1.3)
                print(f"  ⚠ 平台化检测到！难度 {old_diff:.1f} → {self.difficulty:.1f}")

            # 输出
            if g % 5 == 0:
                div_str = ""
                if self.diversity_log and self.diversity_log[-1]['gen'] == g:
                    d = self.diversity_log[-1]
                    div_str = f" | div_mean={d['mean']:.3f}"
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} | "
                      f"fit={ckpt['best_fitness']:.4f}/{ckpt['avg_fitness']:.4f} | "
                      f"pred={ckpt['avg_pred_error']:.4f}/{ckpt['median_pred_error']:.4f}/{ckpt['worst_pred_error']:.4f} | "
                      f"nodes={ckpt['best_nodes']:2d} | lin={ckpt['n_lineages']:2d} "
                      f"sh={ckpt['shannon']:.3f} | dom={ckpt['dominant_pct']:.2f}{div_str}")

        # 保存最终结果
        with open("evolution_log_v8_200g.json", "w", encoding="utf-8") as f:
            json.dump({
                'history': self.history,
                'lineage_log': self.lineage_log,
                'diversity_log': self.diversity_log,
                'num_gens': num_gens,
                'final_difficulty': self.difficulty,
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n保存到 evolution_log_v8_200g.json")

        # 删除断点（完成）
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

        return self.history[-1]


def run_multi_step_prediction_test(world, n_steps_list=None, n_trials=8):
    """
    多步前瞻预测测试——v8 升级：更多步数 + 更严格 baseline。
    """
    if n_steps_list is None:
        n_steps_list = [1, 3, 5, 10, 20, 50]

    print(f"\n{'='*70}")
    print(f"多步前瞻预测测试 (v8)")
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
            for _ in range(50):
                sensory = env.get_sensory()
                test_org.plan(sensory)
                env.step(1)
                actual = env.get_sensory()
                test_org.learn_and_update(actual, 1)

            sensory = env.get_sensory()
            predictions = []
            state_nodes = list(test_org.wm.nodes)
            current_pred = sensory[:]
            for n in range(n_steps):
                pred, state_nodes = test_org.wm_forward_from_nodes(state_nodes, current_pred)
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
        results['evolved'][str(n_steps)] = round(evo_avg, 6)
        results['baseline'][str(n_steps)] = round(base_avg, 6)
        ratio = base_avg / max(1e-8, evo_avg) if evo_avg > 0 else float('inf')
        print(f"  N={n_steps:3d} 步 | 进化={evo_avg:.5f}  baseline={base_avg:.5f}  (baseline/进化={ratio:.1f}x)")

    with open("multi_step_prediction_v8.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("保存到 multi_step_prediction_v8.json")
    return results


if __name__ == "__main__":
    random.seed(42)
    world = World(n_organisms=30, steps_per_gen=400, difficulty=1.0)
    final = world.run(num_gens=200, resume=False)
    run_multi_step_prediction_test(world)
