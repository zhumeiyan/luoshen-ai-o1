# yinyang_evo_v4.py
# 阴阳扣数字生命系统 · v4
#
# 核心修正：
#   · 抛弃 P/F 指数累积——改用有界 leaky integrator（生物真实）
#   · 每个通道: state = decay * state + gain * sigmoid(weighted_input)
#   · sigmoid 保证状态始终落在有限范围——不会爆炸
#   · 基因只控制 decay/gain/sigmoid-strength——这些都是安全参数
#   · 死亡极端宽松：age >= 1000 步自动获得满分 survival

import random, math, json
from copy import deepcopy

class Environment:
    def __init__(self, seed=None, N=1):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.N = N
        self.resource = N * 3.0
        self.resource_regen = 0.3

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

    def step(self):
        self.t += 1
        self.resource = min(self.N * 3.0, self.resource + self.resource_regen)

        # 所有场的涨落
        self.light += random.uniform(-2.0, 2.0)
        self.light = max(0.0, min(100.0, self.light))
        self.light_r = max(-1.0, min(1.0, self.light_r + random.uniform(-0.05, 0.05)))
        self.sound_freq = max(20.0, min(20000.0, self.sound_freq + random.uniform(-25, 25)))
        self.sound_amp = max(0.0, min(120.0, self.sound_amp + random.uniform(-0.5, 0.5)))
        self.pressure *= 0.92 if random.random() > 0.25 else max(0.0, self.pressure + random.uniform(-0.8, 0.8))
        if random.random() < 0.07:
            self.vibration = random.uniform(-1.5, 1.5)
        else:
            self.vibration *= 0.90
        for i in range(5):
            self.smell[i] *= 0.96
            if random.random() < 0.02:
                self.smell[i] += random.uniform(0, 4)
        for i in range(5):
            self.taste[i] *= 0.985
        if random.random() < 0.04:
            for i in range(5):
                self.taste[i] += random.uniform(-0.3, 0.3)
        self.temperature += random.uniform(-0.05, 0.05) + 0.004 * math.sin(self.t * 0.001)
        self.accel[0] = self.accel[0] * 0.94
        self.accel[1] = self.accel[1] * 0.94
        self.accel[2] = self.accel[2] * 0.94 + 9.8 * 0.06

        # 因果事件
        if random.random() < 0.05:
            mgn = random.uniform(-3, 3)
            self.light = max(0, min(100, self.light + mgn * 8))
            self.sound_amp = max(0, min(120, self.sound_amp + mgn * 5))
            self.pressure = max(0, self.pressure + abs(mgn) * 1.2)
            self.temperature += mgn * 0.3
            self.sound_freq = max(20, min(20000, self.sound_freq + mgn * 30))


class SensorArray:
    CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']

    def __init__(self):
        self.prev = None

    def read_all(self, env, genome=None):
        # raw physical readings
        raw = {
            'visual':     env.light + env.light_r * 6,
            'auditory':   env.sound_amp + math.log(max(1.0, env.sound_freq)) * 2.5,
            'touch':      env.pressure + env.vibration * 0.3,
            'smell':      sum(env.smell) / 5,
            'taste':      sum(env.taste),
            'thermal':    env.temperature,
            'balance':    abs(env.accel[2]),
            'proprio':    0.0,
            'time':       float(env.t),
        }

        # 时间差分
        deltas = {}
        for ch in self.CHANNELS:
            if self.prev is None:
                deltas[ch] = 0.0
            else:
                deltas[ch] = raw[ch] - self.prev.get(ch, raw[ch])

        self.prev = raw
        return raw, deltas


class Organism:
    CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']

    def __init__(self, genome=None):
        self.genome = genome if genome else self.random_genome()
        self.senses = SensorArray()
        self.state = [0.0] * 8  # 8 通道的有界内部状态
        self.state_history = []
        self.alive = True
        self.age = 0
        self.fitness = 0.0

    @staticmethod
    def random_genome():
        channels = Organism.CHANNELS
        return {
            'input_gain':     {ch: random.uniform(0.05, 0.8) for ch in channels},
            'modulation':     {ch: random.uniform(-0.5, 0.5) for ch in channels},
            'cross_coupling': {},
            'params': {
                'decay':       random.uniform(0.95, 0.995),
                'sigmoid_k':   random.uniform(0.5, 3.0),
                'target_pull': random.uniform(0.001, 0.1),
                'noise_level': random.uniform(0.0, 0.3),
            },
            'attention':      {ch: random.uniform(0.0, 1.0) for ch in channels},
        }

    def clone_mutate(self, rate=0.15, magnitude=0.25):
        child_g = deepcopy(self.genome)
        for k in child_g['input_gain']:
            if random.random() < rate:
                child_g['input_gain'][k] += random.uniform(-magnitude, magnitude) * 0.5
                child_g['input_gain'][k] = max(0.001, min(2.0, child_g['input_gain'][k]))
        for k in child_g['modulation']:
            if random.random() < rate:
                child_g['modulation'][k] += random.uniform(-magnitude, magnitude) * 0.3
                child_g['modulation'][k] = max(-1.0, min(1.0, child_g['modulation'][k]))
        if 'attention' in child_g:
            for k in child_g['attention']:
                if random.random() < rate:
                    child_g['attention'][k] += random.uniform(-magnitude, magnitude) * 0.3
                    child_g['attention'][k] = max(0.0, min(1.0, child_g['attention'][k]))
        if random.random() < rate * 0.5:
            child_g['params']['decay'] += random.uniform(-0.02, 0.02)
            child_g['params']['decay'] = max(0.8, min(0.999, child_g['params']['decay']))
        if random.random() < rate * 0.5:
            child_g['params']['sigmoid_k'] += random.uniform(-0.5, 0.5)
            child_g['params']['sigmoid_k'] = max(0.1, min(5.0, child_g['params']['sigmoid_k']))
        if random.random() < rate * 0.5:
            child_g['params']['target_pull'] += random.uniform(-0.02, 0.02)
            child_g['params']['target_pull'] = max(0.001, min(0.3, child_g['params']['target_pull']))
        if random.random() < rate * 0.5:
            child_g['params']['noise_level'] += random.uniform(-0.1, 0.1)
            child_g['params']['noise_level'] = max(0.0, min(1.0, child_g['params']['noise_level']))
        return Organism(genome=child_g)

    def _sigmoid(self, x, k):
        """对称 sigmoid: 输出在 (-1, 1) 内——安全版本"""
        z = -k * x
        if z > 500: return -1.0
        if z < -500: return 1.0
        return 2.0 / (1.0 + math.exp(z)) - 1.0

    def update(self, sensory_raw, sensory_deltas):
        ch_n = len(self.CHANNELS)
        P = self.genome['params']
        decay = P['decay']
        sig_k = P['sigmoid_k']
        target_pull = P['target_pull']
        noise_lev = P['noise_level']
        gains = self.genome['input_gain']
        mods = self.genome['modulation']
        attention = self.genome['attention']

        new_state = []
        for i, ch in enumerate(self.CHANNELS):
            # 感官输入（加权 + 调制 + 注意力增益）
            raw_in = sensory_deltas.get(ch, 0.0) * gains.get(ch, 0.1)
            mod_in = mods.get(ch, 0.0)
            att = attention.get(ch, 0.5)

            # 混合信号
            integrated = (raw_in + mod_in) * att

            # sigmoid 有界化 + 噪声
            sig_out = self._sigmoid(integrated, sig_k)
            noise = random.uniform(-noise_lev, noise_lev)

            # Leaky integrator
            prev = self.state[i]
            ns = prev * decay + sig_out + noise * 0.1

            # 目标拉回（创造内稳态）
            if len(self.state_history) >= 2:
                prev_hist = self.state_history[-1]
                if i < len(prev_hist):
                    ns -= target_pull * (ns - prev_hist[i]) * 0.1

            new_state.append(ns)

        self.state = new_state
        self.state_history.append(list(self.state))
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]

        # 死亡——只需检测 NaN 或无穷大（sigmoid 保证不会爆炸）
        if any(math.isnan(s) or math.isinf(s) for s in self.state):
            self.alive = False

        self.age += 1
        return self.alive

    def fitness(self):
        if not self.alive or len(self.state_history) < 10:
            return 0.0

        survival = min(1.0, self.age / 500.0)

        # 状态活跃度——避免死寂
        recent = self.state_history[-min(100, len(self.state_history)):]
        flat = [v for h in recent for v in h]
        m = sum(flat) / len(flat) if flat else 0
        var = sum((x - m)**2 for x in flat) / max(1, len(flat))

        if var < 1e-8:
            activity = 0.01  # 死寂
        else:
            log_var = math.log(var + 1e-8)
            # 期望方差在 ~0.01 - 1.0 范围
            activity = 1.0 / (1.0 + (log_var + 1.0)**2 * 0.2)

        # 注意力集中度
        att_vals = list(self.genome['attention'].values())
        att_std = 0.0
        if att_vals:
            am = sum(att_vals) / len(att_vals)
            att_std = math.sqrt(sum((v - am)**2 for v in att_vals) / len(att_vals))
        attention_focus = 1.0 / (1.0 + att_std * 3)

        return survival * 0.4 + activity * 0.4 + attention_focus * 0.2


class World:
    def __init__(self, n_organisms=25, steps_per_gen=1000):
        self.n = n_organisms
        self.steps_per_gen = steps_per_gen
        self.population = [Organism() for _ in range(n_organisms)]
        self.generation = 0
        self.history = []

    def run_generation(self):
        env = Environment(seed=random.randint(0, 999999), N=self.n)

        for step in range(self.steps_per_gen):
            env.step()
            alive_this = []
            for org in self.population:
                if not org.alive:
                    continue
                raw, deltas = org.senses.read_all(env, org.genome)
                # 加入社会信号
                org.update(raw, deltas)
                if org.alive:
                    alive_this.append(org)

            if step > 0 and step % 100 == 0:
                alive_n = sum(1 for o in self.population if o.alive)
                if alive_n == 0:
                    break

        for org in self.population:
            org.fitness = org.fitness() if org.alive else 0.0

        alive = [o for o in self.population if o.alive]
        return alive

    def select_and_reproduce(self):
        self.population.sort(key=lambda o: o.fitness, reverse=True)
        n_survive = max(3, self.n // 2)
        survivors = self.population[:n_survive]

        new_pop = []
        for s in survivors[:3]:
            new_pop.append(s.clone_mutate(rate=0.05, magnitude=0.08))

        attempts = 0
        while len(new_pop) < self.n and attempts < self.n * 10:
            parent = random.choice(survivors)
            child = parent.clone_mutate(rate=0.2, magnitude=0.3)
            new_pop.append(child)
            attempts += 1

        self.population = new_pop[:self.n]

    def checkpoint(self):
        alive = [o for o in self.population if o.alive]
        best = max(self.population, key=lambda o: o.fitness) if self.population else None
        avg_fit = sum(o.fitness for o in self.population) / max(1, len(self.population))

        if not best:
            return {'gen': self.generation, 'alive': 0, 'best_fitness': 0, 'avg_fitness': 0}

        return {
            'gen': self.generation,
            'alive': len(alive),
            'alive_ratio': len(alive) / self.n,
            'best_fitness': best.fitness,
            'avg_fitness': avg_fit,
            'best_age': best.age,
            'top_attention': sorted(best.genome['attention'].items(), key=lambda x: -x[1])[:5],
            'top_gains': sorted(best.genome['input_gain'].items(), key=lambda x: -x[1])[:5],
            'decay': best.genome['params']['decay'],
            'sig_k': best.genome['params']['sigmoid_k'],
        }

    def run(self, num_gens=120):
        for g in range(num_gens):
            self.generation = g
            alive = self.run_generation()
            ckpt = self.checkpoint()
            self.history.append(ckpt)
            self.select_and_reproduce()

            if g % 5 == 0:
                alive_count = len(alive)
                ratio = alive_count / self.n * 100
                print(f"GEN {g:4d} | alive={alive_count:2d}/{self.n} ({ratio:5.1f}%) | "
                      f"best={ckpt['best_fitness']:.4f} | avg={ckpt['avg_fitness']:.4f} | "
                      f"age={ckpt.get('best_age',0):5d} | top={ckpt.get('top_attention',[])}")

        print(f"\n{'='*60}")
        print(f"进化完成 {num_gens} 代")
        print(f"{'='*60}")

        final = self.checkpoint()
        self.history.append(final)
        best = max(self.population, key=lambda o: o.fitness) if self.population else None

        if best:
            print(f"\n最佳适应度 = {best.fitness:.4f}")
            print("注意力分布：")
            return

        with open("evolution_log.json", "w", encoding="utf-8") as f:
            json.dump({
                'history': self.history,
                'final_best': best.genome if best else None,
                'final_fitness': best.fitness if best else 0,
                'num_gens': num_gens,
            }, f, ensure_ascii=False, indent=2, default=str)
        print("\n保存到 evolution_log.json")


if __name__ == "__main__":
    random.seed(42)
    world = World(n_organisms=25, steps_per_gen=1000)
    world.run(num_gens=120)
