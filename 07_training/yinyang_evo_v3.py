# yinyang_evo_v3.py
# 阴阳扣数字生命系统 · 进化 v3
#
# 修正：
#   · 每个感官通道有独立的 P/F 核心（8 核并行），不再 P=F 死锁
#   · 行动 = 决策下一个"偏置输入"来驱动自身核心
#   · 资源 = 行动准确度的奖励（正反馈）
#   · 死亡条件只有"状态指标爆破"
#   · 完全开放，没有收敛压力

import random, math, json
from copy import deepcopy

class PFCore:
    """独立的阴阳扣 P/F 核心"""
    def __init__(self, p0=0.0, f0=0.0):
        self.P = p0
        self.F = f0

    def step(self, E, bias=0.0):
        # bias: 生物行动带来的外部输入
        f_E = 0.1 * (E + bias)
        g_E = 0.1 * (E - bias * 0.5)  # 不对称 → P ≠ F 可能
        self.P = self.F + f_E
        self.F = self.P + g_E + (self.F - self.P) * 0.02  # 带回差
        return self.P, self.F


class Environment:
    def __init__(self, seed=None, N=1):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.N = N
        self.resource = N * 2.0
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
        self.resource = min(self.N * 2.0, self.resource + self.resource_regen)

        # 基础漂移
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

    def read_all(self, env, genome):
        sw = genome['sense_weights']
        raw = {
            'visual':     (env.light + env.light_r * 6) * sw.get('visual', 1.0),
            'auditory':   (env.sound_amp + math.log(max(1.0, env.sound_freq)) * 2.5) * sw.get('auditory', 1.0),
            'touch':      (env.pressure + env.vibration * 0.3) * sw.get('touch', 1.0),
            'smell':      (sum(env.smell) / 5) * sw.get('smell', 1.0),
            'taste':      (sum(env.taste)) * sw.get('taste', 1.0),
            'thermal':    env.temperature * sw.get('thermal', 1.0),
            'balance':    abs(env.accel[2]) * sw.get('balance', 1.0),
            'proprio':    0.0,
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


class Mirror:
    def observe(self, state_vector, trace):
        if len(trace) < 2:
            return None
        recent = trace[-10:]
        A_prime = -sum(recent) / len(recent)
        diff = state_vector - A_prime
        return {'diff': diff, 'recent_mean': A_prime}


class Organism:
    N_CHANNELS = 8
    CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']

    def __init__(self, genome=None):
        self.genome = genome if genome else self.random_genome()
        self.senses = SensorArray()
        self.cores = [PFCore() for _ in range(self.N_CHANNELS)]
        self.mirror = Mirror()
        self.state_vector = [0.0] * self.N_CHANNELS
        self.combined_state_history = []
        self.mirror_log = []
        self.action_history = []
        self.alive = True
        self.age = 0
        self.fitness = 0.0

    @staticmethod
    def random_genome():
        channels = Organism.CHANNELS
        return {
            # 感官权重——每个物理场被各个通道读取的强度
            'sense_weights': {ch: random.uniform(0.1, 2.0) for ch in channels},
            # 行动策略——每个通道的 P/F 差值如何影响行动偏置
            'action_map': {ch: {'bias': random.uniform(-1.0, 1.0),
                                'sign': random.choice([-1, 1])} for ch in channels},
            # 核心间耦合——通道间互相驱动
            'coupling': {},
            # 调制
            'responsiveness': random.uniform(0.5, 2.0),
            'memory_tau': random.uniform(0.95, 0.999),
            # 内稳
            'target': {ch: random.uniform(-30, 30) for ch in channels},
            'band': {ch: random.uniform(1.0, 15.0) for ch in channels},
        }

    def clone_mutate(self, rate=0.15, magnitude=0.25):
        child_g = deepcopy(self.genome)
        for k in child_g['sense_weights']:
            if random.random() < rate:
                child_g['sense_weights'][k] *= (1 + random.uniform(-magnitude, magnitude))
                child_g['sense_weights'][k] = max(0.01, min(5.0, child_g['sense_weights'][k]))
        for k in child_g['action_map']:
            if random.random() < rate:
                child_g['action_map'][k]['bias'] += random.uniform(-magnitude, magnitude)
                child_g['action_map'][k]['bias'] = max(-3.0, min(3.0, child_g['action_map'][k]['bias']))
        if random.random() < rate * 0.5:
            child_g['responsiveness'] *= (1 + random.uniform(-magnitude, magnitude))
            child_g['responsiveness'] = max(0.1, min(5.0, child_g['responsiveness']))
        if random.random() < rate * 0.5:
            child_g['memory_tau'] += random.uniform(-0.01, 0.01)
            child_g['memory_tau'] = max(0.80, min(0.999, child_g['memory_tau']))
        for k in child_g['target']:
            if random.random() < rate * 0.3:
                child_g['target'][k] += random.uniform(-10, 10)
                child_g['target'][k] = max(-200, min(200, child_g['target'][k]))
        for k in child_g['band']:
            if random.random() < rate * 0.3:
                child_g['band'][k] *= (1 + random.uniform(-magnitude, magnitude))
                child_g['band'][k] = max(0.1, min(50.0, child_g['band'][k]))
        return Organism(genome=child_g)

    def update(self, sensory_raw, sensory_deltas):
        """
        行为循环：感知 → 各通道 P/F 独立演化 → 行动决策 → 核心间耦合
        """
        aw = self.genome['action_map']
        tau = self.genome['memory_tau']
        alpha = self.genome['responsiveness']

        # 每个 P/F 核心独立演化
        action_biases = []
        new_state_vector = []

        for i, ch in enumerate(self.CHANNELS):
            core = self.cores[i]
            E = sensory_deltas.get(ch, 0.0)

            # 行动偏置 = 当前 P/F 差的响应
            pf_diff = core.P - core.F
            bias_from_action = aw.get(ch, {}).get('bias', 0) * pf_diff * 0.01 * alpha

            # 目标追随
            target = self.genome['target'].get(ch, 0)
            band = self.genome['band'].get(ch, 5)
            state_err = target - new_state_vector[i-1] if i > 0 else target - self.state_vector[i]
            if abs(state_err) > band:
                bias_from_action += state_err * 0.005 * alpha

            # P/F 步骤
            p, f = core.step(abs(E), bias_from_action)
            new_state_vector.append(p)

            action_biases.append(bias_from_action)

        # 状态衰减 + 更新
        for i in range(self.N_CHANNELS):
            self.state_vector[i] = self.state_vector[i] * tau + new_state_vector[i] * (1 - tau)

        # 组合状态 = 各通道状态的加权和
        weights = [abs(self.state_vector[i]) for i in range(self.N_CHANNELS)]
        wsum = sum(weights) + 1e-9
        combined = sum(self.state_vector[i] * (weights[i] / wsum) for i in range(self.N_CHANNELS))
        self.combined_state_history.append(combined)

        # 镜面
        m = self.mirror.observe(combined, self.combined_state_history)
        if m:
            self.mirror_log.append(m)

        # 行动日志
        self.action_history.append({
            'step': self.age,
            'biases': action_biases,
            'state': list(self.state_vector),
        })
        if len(self.action_history) > 200:
            self.action_history = self.action_history[-200:]

        # 死亡——极端爆炸
        if len(self.combined_state_history) >= 5 and abs(combined) > 1e9:
            self.alive = False

        self.age += 1
        return self.alive

    def fitness(self):
        if not self.alive or len(self.combined_state_history) < 5:
            return 0.0

        survival = min(1.0, self.age / 1000.0)

        recent = self.combined_state_history[-min(100, len(self.combined_state_history)):]
        m = sum(recent) / len(recent)
        var = sum((x - m)**2 for x in recent) / len(recent)

        if var < 1e-8:
            stability = 0.01
        else:
            log_var = math.log(var + 1e-8)
            stability = 1.0 / (1.0 + (log_var - 1.0)**2 * 0.1)  # 最优方差 ~e

        diversity = 0.0
        if len(self.genome['sense_weights']) >= 2:
            vals = list(self.genome['sense_weights'].values())
            mean_w = sum(vals) / len(vals)
            std_w = math.sqrt(sum((v - mean_w)**2 for v in vals) / len(vals))
            diversity = min(1.0, std_w / 1.5)

        return survival * 0.4 + stability * 0.4 + diversity * 0.2


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

            alive_this_step = []
            for org in self.population:
                if not org.alive:
                    continue
                raw, deltas = org.senses.read_all(env, org.genome)
                org.update(raw, deltas)
                if org.alive:
                    alive_this_step.append(org)

        for org in self.population:
            org.fitness = org.fitness() if org.alive else 0.0

        n_alive = len([o for o in self.population if o.alive])
        if n_alive == 0 and self.population:
            # 全死了——复活最强的那一个，让进化不中断
            self.population[0].alive = True
            self.population[0].fitness = 0.001

        return [o for o in self.population if o.alive]

    def select_and_reproduce(self):
        self.population.sort(key=lambda o: o.fitness, reverse=True)
        n_survive = max(3, self.n // 2)
        survivors = self.population[:n_survive]

        new_pop = []
        # 精英保留前 3 个不变异
        for s in survivors[:3]:
            new_pop.append(s.clone_mutate(rate=0.05, magnitude=0.08))

        # 突变繁殖填满
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

        sw = best.genome['sense_weights']
        top_ch = sorted(sw.items(), key=lambda x: -x[1])[:5]

        return {
            'gen': self.generation,
            'alive': len(alive),
            'alive_ratio': len(alive) / self.n,
            'best_fitness': best.fitness,
            'avg_fitness': avg_fit,
            'best_age': best.age,
            'top_senses': top_ch,
            'responsiveness': best.genome['responsiveness'],
            'memory_tau': best.genome['memory_tau'],
            'state_range': (min(best.combined_state_history[-50:]) if len(best.combined_state_history) > 50 else 0,
                            max(best.combined_state_history[-50:]) if len(best.combined_state_history) > 50 else 0),
        }

    def run(self, num_gens=150):
        for g in range(num_gens):
            self.generation = g
            alive = self.run_generation()
            ckpt = self.checkpoint()
            self.history.append(ckpt)
            self.select_and_reproduce()

            if g % 5 == 0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} ({ckpt.get('alive_ratio',0)*100:5.1f}%) | "
                      f"best={ckpt['best_fitness']:.4f} | avg={ckpt['avg_fitness']:.4f} | "
                      f"age={ckpt.get('best_age',0):5d} | senses={ckpt.get('top_senses',[])}")

        # 最终
        print(f"\n{'='*60}")
        print(f"进化完成 {num_gens} 代")
        print(f"{'='*60}")

        final_ckpt = self.checkpoint()
        self.history.append(final_ckpt)
        best = max(self.population, key=lambda o: o.fitness) if self.population else None

        if best:
            print("\n最佳个体基因：")
            print("  感官权重：")
            for ch, w in sorted(best.genome['sense_weights'].items(), key=lambda x: -x[1]):
                print(f"    {ch:12s} = {w:.4f}")
            print("  行动策略：")
            for ch, v in sorted(best.genome['action_map'].items(), key=lambda x: -abs(x[1]['bias'])):
                print(f"    {ch:12s}  bias={v['bias']:+.4f}")
            print(f"  responsiveness = {best.genome['responsiveness']:.4f}")
            print(f"  memory_tau     = {best.genome['memory_tau']:.4f}")
            print(f"  存活步数       = {best.age}")
            print(f"  适应度         = {best.fitness:.4f}")

            if best.combined_state_history:
                recent = best.combined_state_history[-20:]
                print(f"  组合状态尾20   = {[f'{x:.2f}' for x in recent]}")

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
    world.run(num_gens=150)
