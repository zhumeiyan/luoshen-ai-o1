# yinyang_evo_v5.py
# 阴阳扣数字生命系统 · v5
#
# v4 全部死亡的根因修复：
#   environment line 44 operator precedence bug — pressure 场平方爆炸 →
#   raw[touch] = inf → sigmoid(z=±inf) = NaN → 所有生物第一通道 NaN 死亡。
#
# 架构重设计：
#   · 环境：完全有界的物理场，无爆炸风险
#   · 生物：有界 leaky integrator + sigmoid 双保险
#   · 选择压力：能量代谢 — 从环境摄取，消耗代谢成本
#   · 死亡：能量耗尽（真正的淘汰）
#   · 繁殖：只有能量充裕者能繁殖（真正的保留）
#   · 感官：完全从环境场生长，代码无硬编码感官知识

import random, math, json
from copy import deepcopy


class Environment:
    """完全有界环境——所有场值都在有限范围内"""
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.resource = 10.0           # 全局资源池

        # 9 个人类感官对应的物理场（完全有界）
        self.light = 50.0             # 0-100
        self.light_r = 0.0            # -1 ~ 1
        self.sound_freq = 440.0       # 20-20000
        self.sound_amp = 30.0         # 0-120
        self.pressure = 0.0           # 0-20
        self.vibration = 0.0          # -2 ~ 2
        self.smell = [0.0]*5          # 每维 0-10
        self.taste = [0.0]*5          # 每维 -5~5
        self.temperature = 20.0       # 15-30（有界）
        self.accel = [0.0, 0.0, 9.8]  # 每维 -2~12

    def step(self):
        self.t += 1
        self.resource = min(20.0, self.resource + 0.15)

        # 光：有界随机游走
        self.light += random.uniform(-2.0, 2.0)
        self.light = max(0.0, min(100.0, self.light))
        self.light_r = max(-1.0, min(1.0, self.light_r + random.uniform(-0.05, 0.05)))

        # 声：频率、振幅 有界游走
        self.sound_freq = max(20.0, min(20000.0, self.sound_freq + random.uniform(-25, 25)))
        self.sound_amp = max(0.0, min(120.0, self.sound_amp + random.uniform(-0.5, 0.5)))

        # 压力：正确的 if/else 分支——75% 衰减, 25% 增量（修复 v4 的爆炸 bug）
        if random.random() > 0.25:
            self.pressure *= 0.92
        else:
            self.pressure = max(0.0, self.pressure + random.uniform(-0.8, 0.8))
        self.pressure = min(20.0, self.pressure)

        # 振动
        if random.random() < 0.07:
            self.vibration = random.uniform(-1.5, 1.5)
        else:
            self.vibration *= 0.90

        # 气味：5 个独立维度
        for i in range(5):
            self.smell[i] *= 0.96
            self.smell[i] = min(10.0, max(0.0, self.smell[i]))
            if random.random() < 0.02:
                self.smell[i] += random.uniform(0, 4)
                self.smell[i] = min(10.0, self.smell[i])

        # 味觉：5 个独立维度
        for i in range(5):
            self.taste[i] *= 0.985
            if random.random() < 0.04:
                self.taste[i] += random.uniform(-0.3, 0.3)
            self.taste[i] = max(-5.0, min(5.0, self.taste[i]))

        # 温度：有界游走（有昼夜周期）
        self.temperature += random.uniform(-0.05, 0.05) + 0.004 * math.sin(self.t * 0.001)
        self.temperature = max(15.0, min(35.0, self.temperature))

        # 加速度：阻尼趋于 9.8
        self.accel[0] = max(-2.0, min(12.0, self.accel[0] * 0.94))
        self.accel[1] = max(-2.0, min(12.0, self.accel[1] * 0.94))
        self.accel[2] = max(-2.0, min(12.0, self.accel[2] * 0.94 + 9.8 * 0.06))

        # 因果事件（少见）
        if random.random() < 0.05:
            mgn = random.uniform(-3, 3)
            self.light = max(0, min(100, self.light + mgn * 8))
            self.sound_amp = max(0, min(120, self.sound_amp + mgn * 5))
            self.pressure = min(20.0, self.pressure + abs(mgn) * 1.2)
            self.temperature = max(15, min(35, self.temperature + mgn * 0.3))
            self.sound_freq = max(20, min(20000, self.sound_freq + mgn * 30))


class SensorArray:
    """感官层：将物理场转为 8 通道生物可读形式"""
    CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']

    def __init__(self):
        self.prev = None

    def read_all(self, env, genome=None):
        raw = {
            'visual':   env.light + env.light_r * 6,                # 光强度 + 颜色偏差
            'auditory': env.sound_amp + math.log(max(1.0, env.sound_freq)) * 2.5,
            'touch':    env.pressure + env.vibration * 0.3,
            'smell':    sum(env.smell) / 5,
            'taste':    sum(env.taste),
            'thermal':  env.temperature,
            'balance':  abs(env.accel[2] - 9.8),   # 偏离重力 = 失衡
            'proprio':  0.0,                       # 本体感觉（暂为 0，由生物自身填充）
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
    CHANNELS = SensorArray.CHANNELS

    def __init__(self, genome=None):
        self.genome = genome if genome else self.random_genome()
        self.senses = SensorArray()
        self.state = [0.0] * 8
        self.state_history = []
        self.alive = True
        self.age = 0
        self.energy = 5.0             # 基础能量

    @staticmethod
    def random_genome():
        channels = Organism.CHANNELS
        return {
            'input_gain': {ch: random.uniform(0.1, 1.5) for ch in channels},
            'modulation': {ch: random.uniform(-0.3, 0.3) for ch in channels},
            'params': {
                'decay':       random.uniform(0.85, 0.99),
                'sigmoid_k':   random.uniform(0.3, 2.5),
                'metabolic':   random.uniform(0.02, 0.15),   # 代谢率——越小越节能
            },
            'attention':  {ch: random.uniform(0.0, 1.0) for ch in channels},
        }

    def clone_mutate(self, rate=0.15, magnitude=0.25):
        child_g = deepcopy(self.genome)
        for k in child_g['input_gain']:
            if random.random() < rate:
                child_g['input_gain'][k] += random.uniform(-magnitude, magnitude) * 0.5
                child_g['input_gain'][k] = max(0.01, min(3.0, child_g['input_gain'][k]))
        for k in child_g['modulation']:
            if random.random() < rate:
                child_g['modulation'][k] += random.uniform(-magnitude, magnitude) * 0.3
                child_g['modulation'][k] = max(-1.0, min(1.0, child_g['modulation'][k]))
        if 'attention' in child_g:
            for k in child_g['attention']:
                if random.random() < rate:
                    child_g['attention'][k] += random.uniform(-magnitude, magnitude) * 0.3
                    child_g['attention'][k] = max(0.0, min(1.0, child_g['attention'][k]))
        if random.random() < rate:
            child_g['params']['decay'] += random.uniform(-0.02, 0.02)
            child_g['params']['decay'] = max(0.7, min(0.995, child_g['params']['decay']))
        if random.random() < rate:
            child_g['params']['sigmoid_k'] += random.uniform(-0.3, 0.3)
            child_g['params']['sigmoid_k'] = max(0.1, min(4.0, child_g['params']['sigmoid_k']))
        if random.random() < rate:
            child_g['params']['metabolic'] += random.uniform(-0.01, 0.01)
            child_g['params']['metabolic'] = max(0.005, min(0.3, child_g['params']['metabolic']))
        return Organism(genome=child_g)

    def _safe_sigmoid(self, x, k):
        """安全 sigmoid：输入任何数都返回 (-1, 1)，永远不产生 NaN/inf"""
        if math.isnan(x) or math.isinf(x):
            return 0.0
        z = -k * x
        if z > 500: return -1.0
        if z < -500: return 1.0
        try:
            return 2.0 / (1.0 + math.exp(z)) - 1.0
        except (OverflowError, ZeroDivisionError):
            return 0.0

    def update(self, sensory_raw, sensory_deltas, env_resource):
        if not self.alive:
            return False

        P = self.genome['params']
        decay = P['decay']
        sig_k = P['sigmoid_k']
        metabolic = P['metabolic']
        gains = self.genome['input_gain']
        mods = self.genome['modulation']
        attention = self.genome['attention']

        new_state = []
        for i, ch in enumerate(self.CHANNELS):
            raw_in = sensory_deltas.get(ch, 0.0) * gains.get(ch, 0.1)
            mod_in = mods.get(ch, 0.0)
            att = attention.get(ch, 0.5)

            # 有界混合 + sigmoid
            integrated = (raw_in + mod_in) * att
            sig_out = self._safe_sigmoid(integrated, sig_k)
            noise = random.uniform(-0.05, 0.05)

            prev = self.state[i]
            ns = prev * decay + sig_out + noise

            # 软拉回——防止状态狂拉着 Silence（轻微惯性）
            ns *= 0.999

            new_state.append(ns)

        self.state = new_state
        self.state_history.append(list(self.state))
        if len(self.state_history) > 500:
            self.state_history = self.state_history[-500:]

        # 能量代谢——消耗
        self.energy -= metabolic

        # 能量摄取——成功感知环境变化时获得奖励
        # 感知 "有意义" 的变化（delta 大）→ 获得能量
        recent_delta_mag = sum(abs(sensory_deltas.get(ch, 0.0)) for ch in self.CHANNELS) / len(self.CHANNELS)
        # 感知越精准（状态跟上变化）→ 奖励越高
        if len(self.state_history) >= 2:
            state_change = sum(abs(self.state[i] - self.state_history[-2][i]) for i in range(8)) / 8
            # 状态跟上环境 delta → 摄取能量
            tracking_quality = 1.0 / (1.0 + abs(state_change - recent_delta_mag) * 10)
            self.energy += tracking_quality * 0.05

        # 死亡——仅能量耗尽
        if self.energy <= 0:
            self.alive = False

        self.age += 1
        return self.alive

    def can_reproduce(self):
        """只有能量充裕且年龄足够才能繁殖"""
        return self.age >= 50 and self.energy > 3.0

    def compute_fitness(self):
        if not self.alive:
            return 0.0

        # 存活得分
        survival = min(1.0, self.age / 300.0)

        # 能量管理得分
        energy_score = min(1.0, self.energy / 10.0)

        # 状态活跃度
        if len(self.state_history) >= 20:
            recent = self.state_history[-50:]
            flat = [v for h in recent for v in h]
            m = sum(flat) / len(flat)
            var = sum((x - m)**2 for x in flat) / max(1, len(flat))
            if var < 1e-8:
                activity = 0.0
            else:
                log_var = math.log(var + 1e-8)
                activity = 1.0 / (1.0 + (log_var + 1.0)**2 * 0.2)
        else:
            activity = 0.0

        # 注意力聚焦
        att_vals = list(self.genome['attention'].values())
        if att_vals:
            am = sum(att_vals) / len(att_vals)
            att_std = math.sqrt(sum((v - am)**2 for v in att_vals) / len(att_vals))
            attention_score = 1.0 / (1.0 + att_std * 3)
        else:
            attention_score = 0.0

        return survival * 0.3 + energy_score * 0.3 + activity * 0.25 + attention_score * 0.15


class World:
    def __init__(self, n_organisms=25, steps_per_gen=500):
        self.n = n_organisms
        self.steps_per_gen = steps_per_gen
        self.population = [Organism() for _ in range(n_organisms)]
        self.generation = 0
        self.history = []

    def run_generation(self):
        env = Environment(seed=random.randint(0, 999999))

        for step in range(self.steps_per_gen):
            env.step()
            alive_this = []
            for org in self.population:
                if not org.alive:
                    continue
                raw, deltas = org.senses.read_all(env, org.genome)
                org.update(raw, deltas, env.resource)
                if org.alive:
                    alive_this.append(org)

        alive = [o for o in self.population if o.alive]
        return alive

    def select_and_reproduce(self):
        self.population.sort(key=lambda o: o.compute_fitness(), reverse=True)
        n_survive = max(4, self.n // 3)
        survivors = self.population[:n_survive]

        # 克隆最佳
        new_pop = []
        survivor_fits = [(s, s.compute_fitness()) for s in survivors]
        survivor_fits.sort(key=lambda x: x[1], reverse=True)
        for s, sf in survivor_fits[:3]:
            if sf > 0.05:
                new_pop.append(s.clone_mutate(rate=0.05, magnitude=0.08))

        # 交配/变异填充种群
        attempts = 0
        while len(new_pop) < self.n and attempts < self.n * 8:
            if survivors:
                parent = random.choice(survivors[:n_survive])
                child = parent.clone_mutate(rate=0.15, magnitude=0.2)
                new_pop.append(child)
            else:
                new_pop.append(Organism())
            attempts += 1

        self.population = new_pop[:self.n]

    def checkpoint(self):
        alive = [o for o in self.population if o.alive]
        best = max(self.population, key=lambda o: o.compute_fitness()) if self.population else None
        avg_fit = sum(o.compute_fitness() for o in self.population) / max(1, len(self.population))
        avg_energy = sum(o.energy for o in self.population) / max(1, len(self.population))
        avg_age = sum(o.age for o in self.population) / max(1, len(self.population))

        if not best:
            return {'gen': self.generation, 'alive': 0, 'best_fitness': 0}

        return {
            'gen': self.generation,
            'alive': len(alive),
            'alive_ratio': len(alive) / self.n,
            'best_fitness': best.compute_fitness(),
            'avg_fitness': avg_fit,
            'avg_energy': avg_energy,
            'avg_age': avg_age,
            'best_age': best.age,
            'best_energy': best.energy,
            'top_attention': sorted(best.genome['attention'].items(), key=lambda x: -x[1])[:5],
            'decay': best.genome['params']['decay'],
            'metabolic': best.genome['params']['metabolic'],
        }

    def run(self, num_gens=100):
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
                      f"age={ckpt.get('avg_age',0):6.1f} | energy={ckpt.get('avg_energy',0):.3f} | "
                      f"top={ckpt.get('top_attention',[])}")

        print(f"\n{'='*60}")
        print(f"进化完成 {num_gens} 代")
        print(f"{'='*60}")

        # history[-1] = 最后一个 gen 的真实 checkpoint（select/reproduce 之前）
        final_ckpt = self.history[-1] if self.history else None

        if final_ckpt:
            print(f"\n最终代数 (gen {final_ckpt['gen']}) 统计:")
            print(f"  存活率 = {final_ckpt['alive_ratio']*100:.1f}%")
            print(f"  最佳适应度 = {final_ckpt['best_fitness']:.4f}")
            print(f"  平均适应度 = {final_ckpt['avg_fitness']:.4f}")
            print(f"  平均能量 = {final_ckpt['avg_energy']:.3f}")
            print(f"  平均年龄 = {final_ckpt['avg_age']:.1f}")
            print(f"  最佳注意力分布: {final_ckpt.get('top_attention', [])[:5]}")
            print(f"  最佳代谢 = {final_ckpt.get('metabolic', 'N/A')}")
            print(f"  最佳衰减 = {final_ckpt.get('decay', 'N/A')}")

        with open("evolution_log_v5.json", "w", encoding="utf-8") as f:
            json.dump({
                'history': self.history,
                'num_gens': num_gens,
            }, f, ensure_ascii=False, indent=2, default=str)
        print("\n保存到 evolution_log_v5.json")


if __name__ == "__main__":
    random.seed(42)
    world = World(n_organisms=25, steps_per_gen=500)
    world.run(num_gens=100)
