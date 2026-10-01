# yinyang_life_full_evo.py
# 阴阳扣数字生命系统 · 纯进化版
#
# 零判据、零人为干预、零阈值。
# 只做一件事：变异→选择→保留→繁殖→迭代。
# 能活下来的就是对的。不需要代码说它对不对。

import random, math, json
from copy import deepcopy


# ===== 底层 YinyangCore =====
class YinyangCore:
    def __init__(self):
        self.P = 0.0
        self.F = 0.0
        self.t = 0

    def step(self, E):
        self.P = self.F + 0.1 * E
        self.F = self.P + 0.1 * E
        self.t += 1


# ===== 环境 =====
class Environment:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.energy = 100.0
        self.light = 50.0
        self.light_r = 0.0
        self.sound_freq = 440.0
        self.sound_amp = 30.0
        self.pressure = 0.0
        self.vibration = 0.0
        self.smell = [0.0]*5
        self.taste = [0.0]*5  # sweet, sour, bitter, salty, umami
        self.temperature = 20.0
        self.accel = [0.0, 0.0, 9.8]

    def step(self):
        self.t += 1
        # 基础漂移
        self.light += random.uniform(-2.0, 2.0)
        if self.light > 100: self.light = 100.0
        if self.light < 0: self.light = 0.0
        self.light_r += random.uniform(-0.05, 0.05)
        if self.light_r > 1: self.light_r = 1.0
        if self.light_r < -1: self.light_r = -1.0

        self.sound_freq += random.uniform(-25, 25)
        if self.sound_freq > 20000: self.sound_freq = 20000.0
        if self.sound_freq < 20: self.sound_freq = 20.0
        self.sound_amp += random.uniform(-0.5, 0.5)
        if self.sound_amp > 120: self.sound_amp = 120.0
        if self.sound_amp < 0: self.sound_amp = 0.0

        if random.random() < 0.25:
            self.pressure += random.uniform(-0.8, 0.8)
            if self.pressure < 0: self.pressure = 0.0
        else:
            self.pressure *= 0.92

        if random.random() < 0.07:
            self.vibration = random.uniform(-1.5, 1.5)
        else:
            self.vibration *= 0.91

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

        self.accel[0] = self.accel[0] * 0.94 + random.uniform(-0.05, 0.05)
        self.accel[1] = self.accel[1] * 0.94 + random.uniform(-0.05, 0.05)
        self.accel[2] = self.accel[2] * 0.94 + 9.8 * 0.06 + random.uniform(-0.05, 0.05)

        # 因果事件
        if random.random() < 0.06:
            mgn = random.uniform(-3, 3)
            self.light += mgn * 8
            if self.light > 100: self.light = 100.0
            if self.light < 0: self.light = 0.0
            self.sound_amp += mgn * 4
            if self.sound_amp > 120: self.sound_amp = 120.0
            if self.sound_amp < 0: self.sound_amp = 0.0
            self.pressure += abs(mgn) * 1.2
            if self.pressure > 100: self.pressure = 100.0
            self.temperature += mgn * 0.3
            self.sound_freq += mgn * 30
            if self.sound_freq > 20000: self.sound_freq = 20000.0
            if self.sound_freq < 20: self.sound_freq = 20.0

        # 能量
        self.energy -= 0.005
        if self.energy < 0: self.energy = 0.0


class SensorArray:
    """感官读取——物理场 → 原始信号"""
    def read_all(self, env):
        return {
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


class Mirror:
    def observe(self, inner_state, trace):
        if len(trace) < 2:
            return None
        recent = trace[-10:]
        A_prime = -sum(recent) / len(recent)
        return {'A': inner_state, 'A_prime': A_prime, 'diff': inner_state - A_prime}


class Organism:
    """
    一个完整数字生命个体。
    没有预设阈值，没有人为觉醒标志。
    它的生死由环境决定：不能在环境中稳定状态的就死，能稳定遗传的就繁殖。
    """
    def __init__(self, genome=None):
        self.genome = genome if genome else self.random_genome()
        self.state = 0.0
        self.trace = []
        self.mirror_log = []
        self.sensory_log = []
        self.prev_sensory = None
        self.fitness = 0.0
        self.alive = True

    @staticmethod
    def random_genome():
        channels = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']
        genome = {
            'channel_weight': {ch: random.uniform(0.2, 1.5) for ch in channels},
            'channel_bias':   {ch: random.uniform(-0.5, 0.5) for ch in channels},
            'assoc':          {},
            'meta': {
                'integration_rate': random.uniform(0.005, 0.015),
                'decay_rate':       random.uniform(0.995, 0.9995),
                'proprio_gain':     random.uniform(0.1, 1.0),
            }
        }
        for a in channels:
            for b in channels:
                if a < b:
                    genome['assoc'][(a,b)] = random.uniform(-0.3, 0.3)
        return genome

    def clone_with_mutation(self, mutation_rate=0.1, mutation_magnitude=0.2):
        child_genome = deepcopy(self.genome)
        for ch in child_genome['channel_weight']:
            if random.random() < mutation_rate:
                child_genome['channel_weight'][ch] += random.uniform(-mutation_magnitude, mutation_magnitude)
                child_genome['channel_weight'][ch] = max(0.05, min(3.0, child_genome['channel_weight'][ch]))
        for ch in child_genome['channel_bias']:
            if random.random() < mutation_rate:
                child_genome['channel_bias'][ch] += random.uniform(-mutation_magnitude, mutation_magnitude)
                child_genome['channel_bias'][ch] = max(-1.5, min(1.5, child_genome['channel_bias'][ch]))
        for pair in child_genome['assoc']:
            if random.random() < mutation_rate:
                child_genome['assoc'][pair] += random.uniform(-mutation_magnitude, mutation_magnitude)
                child_genome['assoc'][pair] = max(-1.0, min(1.0, child_genome['assoc'][pair]))
        # 元参数带边界变异
        if random.random() < mutation_rate * 0.5:
            child_genome['meta']['integration_rate'] += random.uniform(-0.003, 0.003)
            child_genome['meta']['integration_rate'] = max(0.001, min(0.05, child_genome['meta']['integration_rate']))
        if random.random() < mutation_rate * 0.5:
            child_genome['meta']['decay_rate'] += random.uniform(-0.001, 0.001)
            child_genome['meta']['decay_rate'] = max(0.99, min(0.9999, child_genome['meta']['decay_rate']))
        if random.random() < mutation_rate * 0.5:
            child_genome['meta']['proprio_gain'] += random.uniform(-0.1, 0.1)
            child_genome['meta']['proprio_gain'] = max(0.01, min(2.0, child_genome['meta']['proprio_gain']))
        return Organism(genome=child_genome)

    def update(self, sensory, P, F, mirror_obj):
        channels = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']

        # 时间差分
        deltas = {}
        for ch in channels:
            if self.prev_sensory is None:
                deltas[ch] = 0.0
            else:
                deltas[ch] = sensory.get(ch, 0.0) - self.prev_sensory.get(ch, 0.0)

        # proprioception = state 自身变化
        if len(self.trace) >= 2:
            proprio = (self.trace[-1] - self.trace[-2]) * self.genome['meta']['proprio_gain']
            deltas['proprio'] = proprio * 0.001  # 缩放到可比较范围

        # 加权 delta
        total_delta = 0.0
        for ch in channels:
            w = self.genome['channel_weight'][ch]
            b = self.genome['channel_bias'][ch]
            total_delta += w * (deltas.get(ch, 0.0) + b * 0.01)

        # 关联调制：同向通道对互相增强
        for (a, b), w in self.genome['assoc'].items():
            if a in deltas and b in deltas:
                if deltas[a] * deltas[b] > 0:
                    total_delta += w * deltas[a] * deltas[b] * 0.001

        rate = self.genome['meta']['integration_rate']
        decay = self.genome['meta']['decay_rate']

        # 状态更新
        self.state = self.state * decay + total_delta * rate
        self.trace.append(self.state)
        if len(self.trace) > 500:
            self.trace = self.trace[-500:]

        m = mirror_obj.observe(self.state, self.trace)
        if m:
            self.mirror_log.append(m)
            if len(self.mirror_log) > 200:
                self.mirror_log = self.mirror_log[-200:]

        self.prev_sensory = {k: v for k, v in sensory.items()}
        if len(self.sensory_log) > 5:
            self.sensory_log.pop(0)
        self.sensory_log.append({k: v for k, v in sensory.items()})

        return self.state

    def fitness_score(self):
        if len(self.trace) < 50:
            return 0.001

        recent = self.trace[-50:]
        mean = sum(recent) / len(recent)
        var = sum((x - mean)**2 for x in recent) / len(recent)

        # 适应度 = 在环境中能维持非零非爆炸的状态
        # 对数方差接近某个正值 = 活得好
        if var < 1e-6:
            return 0.001  # 死寂
        if var > 1e8:
            return 0.001  # 爆炸

        # 存活代数
        survival_bonus = min(1.0, len(self.trace) / 500)

        # 状态稳定性
        log_var = math.log(var + 1e-6)
        stability = 1.0 / (1.0 + (log_var - 2.0)**2 * 0.1)  # 最优方差 ~= e^2 ~= 7.4

        # 感应丰富度
        unique_sensory = set()
        for s in self.sensory_log:
            for k, v in s.items():
                if k != 'time':
                    bucket = round(v / 5)
                    unique_sensory.add((k, bucket))
        richness = min(1.0, len(unique_sensory) / 30)

        return stability * 0.5 + survival_bonus * 0.3 + richness * 0.2


# ===== 进化引擎 =====
class Evolution:
    """纯达尔文过程——不做任何干预，只跑变异+选择+繁殖"""
    def __init__(self, population_size=12, steps_per_gen=500):
        self.population_size = population_size
        self.steps_per_gen = steps_per_gen
        self.population = [Organism() for _ in range(population_size)]
        self.env = None
        self.senses = SensorArray()
        self.mirror = Mirror()
        self.core = YinyangCore()
        self.gen = 0
        self.history = []

    def run_generation(self):
        """跑一代——所有个体在同一个环境中竞争"""
        self.env = Environment(seed=random.randint(0, 99999))
        self.core = YinyangCore()
        alive_orgs = list(self.population)

        for step in range(self.steps_per_gen):
            if not alive_orgs:
                break
            self.env.step()
            sensory = self.senses.read_all(self.env)
            E = sum(abs(v) for k, v in sensory.items() if k != 'time') / 8
            self.core.step(E)
            P, F = self.core.P, self.core.F

            step_alive = []
            for org in alive_orgs:
                org.update(sensory, P, F, self.mirror)
                # 死亡条件——状态爆炸
                if len(org.trace) >= 50 and abs(org.trace[-1]) > 1e8:
                    org.alive = False
                    continue
                step_alive.append(org)

            alive_orgs = step_alive

            if self.env.energy <= 0:
                break

        # 全部死亡则复活最高分的一个（保底进化）
        if not alive_orgs and self.population:
            self.population[0].alive = True
            alive_orgs = [self.population[0]]

        # 计算适应度
        for org in self.population:
            org.fitness = org.fitness_score() if org.alive else 0.0

        return alive_orgs

    def select_and_reproduce(self):
        """选择——按适应度排序，top half 繁殖，bottom half 淘汰"""
        self.population.sort(key=lambda o: o.fitness, reverse=True)
        survivors = self.population[:self.population_size // 2]

        new_population = []
        for parent in survivors:
            # 每一个幸存者繁殖 2 个后代（带变异）
            child1 = parent.clone_with_mutation(mutation_rate=0.15, mutation_magnitude=0.2)
            child2 = parent.clone_with_mutation(mutation_rate=0.25, mutation_magnitude=0.4)
            # 自我复制（保留自身基因型）
            parent_copy = parent.clone_with_mutation(mutation_rate=0.05, mutation_magnitude=0.05)
            new_population.extend([child1, child2, parent_copy])

        # 截断到目标种群大小（保留适应度最高的）
        self.population = new_population[:self.population_size]

    def checkpoint(self):
        """记录本代最佳个体的基因型和统计"""
        best = max(self.population, key=lambda o: o.fitness)
        channels = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']
        active_ch = sum(1 for ch in channels if best.genome['channel_weight'][ch] > 1.0)
        var = 0.0
        if len(best.trace) >= 50:
            recent = best.trace[-50:]
            mean = sum(recent)/len(recent)
            var = sum((x-mean)**2 for x in recent)/len(recent)

        return {
            'gen': self.gen,
            'best_fitness': best.fitness,
            'avg_fitness': sum(o.fitness for o in self.population) / len(self.population),
            'active_channels': active_ch,
            'state_variance': var,
            'state_mean': sum(best.trace[-10:]) / 10 if best.trace else 0,
            'genome_sample_ch': list(best.genome['channel_weight'].items())[:4],
            'decay_rate': best.genome['meta']['decay_rate'],
            'integration_rate': best.genome['meta']['integration_rate'],
        }

    def run(self, num_generations=100):
        """跑完整的进化——不做任何干预，只看结果"""
        for g in range(num_generations):
            self.gen = g
            alive = self.run_generation()
            ckpt = self.checkpoint()  # 先 checkpoint（当前 best 还在）
            self.select_and_reproduce()  # 再繁殖出下一代
            self.history.append(ckpt)

            if g % 10 == 0:
                alive_count = len(alive)
                print(f"GEN {g:3d} | alive={alive_count:2d}/{self.population_size} | "
                      f"best_fit={ckpt['best_fitness']:.4f} | "
                      f"active_ch={ckpt['active_channels']} | "
                      f"state_var={ckpt['state_variance']:.4f} | "
                      f"state_mean={ckpt['state_mean']:.2f}")

        # 最终报告
        print(f"\n{'='*60}")
        print(f"进化完成 {num_generations} 代")
        print(f"{'='*60}")

        best = max(self.population, key=lambda o: o.fitness)
        print("\n最佳个体基因（通道权重）：")
        for ch, w in sorted(best.genome['channel_weight'].items(), key=lambda x: -abs(x[1])):
            b = best.genome['channel_bias'][ch]
            print(f"  {ch:10s}  weight={w:+.3f}  bias={b:+.3f}")
        print(f"\n元参数：")
        for k, v in best.genome['meta'].items():
            print(f"  {k:20s} = {v:.5f}")
        print(f"\n适应度 = {best.fitness:.4f}")
        print(f"最终 state 序列尾部：{[f'{x:.2f}' for x in best.trace[-5:]]}")

        # 进化历史
        with open("evolution_log.json", "w", encoding="utf-8") as f:
            json.dump({
                'history': self.history,
                'final_best_genome': {
                    'channel_weight': best.genome['channel_weight'],
                    'channel_bias': best.genome['channel_bias'],
                    'meta': best.genome['meta'],
                },
                'final_fitness': best.fitness,
                'num_gens': num_generations,
            }, f, ensure_ascii=False, indent=2, default=str)

        print("\n进化历史已保存到 evolution_log.json")
        return best


if __name__ == "__main__":
    random.seed(42)
    evo = Evolution(population_size=12, steps_per_gen=500)
    winner = evo.run(num_generations=100)
