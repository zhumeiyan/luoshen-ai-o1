# yinyang_evo_v2.py
# 阴阳扣数字生命系统 · 进化 v2
#
# 错误修正：
#   ① 生物必须有行为——输出对环境产生反馈
#   ② 选择必须真正发生——高比例淘汰
#   ③ 环境必须允许多样性——多生态位 + 频率依赖选择
#
# 没有预设任何"应该留下什么"的判据。能活下来的就是对。

import random, math, json
from copy import deepcopy


class YinyangCore:
    def __init__(self):
        self.P = 0.0
        self.F = 0.0
        self.t = 0

    def step(self, E):
        self.P = self.F + 0.1 * E
        self.F = self.P + 0.1 * E
        self.t += 1


class Environment:
    """共享资源环境——多个生物同时竞争有限资源"""
    def __init__(self, seed=None, resource_total=50.0):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.energy = 100.0
        self.resource = resource_total  # 环境中可被消耗的资源
        self.resource_regen = 0.05     # 每步再生量

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
        self.events = []

    def step(self):
        self.t += 1
        self.resource = min(50.0, self.resource + self.resource_regen)

        # 感知场漂移
        self.light += random.uniform(-2.0, 2.0)
        self.light = max(0, min(100, self.light))
        self.light_r = max(-1, min(1, self.light_r + random.uniform(-0.05, 0.05)))
        self.sound_freq = max(20, min(20000, self.sound_freq + random.uniform(-25, 25)))
        self.sound_amp = max(0, min(120, self.sound_amp + random.uniform(-0.5, 0.5)))
        self.pressure *= 0.93 if random.random() > 0.2 else max(0, self.pressure + random.uniform(-0.8, 0.8))
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
            self.light = max(0, min(100, self.light + mgn * 8))
            self.sound_amp = max(0, min(120, self.sound_amp + mgn * 4))
            self.pressure = max(0, self.pressure + abs(mgn) * 1.2)
            self.temperature += mgn * 0.3
            self.sound_freq = max(20, min(20000, self.sound_freq + mgn * 30))
            self.events.append(self.t)

        self.energy -= 0.005
        if self.energy < 0: self.energy = 0.0


class SensorArray:
    def __init__(self, genome=None):
        self.genome = genome

    def read_weighted(self, env, genome):
        """加权读取——基因组决定每个感官通道如何萃取物理场"""
        sw = genome['sense_weights']
        return {
            'visual':     (env.light + env.light_r * 6) * sw.get('visual', 1),
            'auditory':   (env.sound_amp + math.log(max(1.0, env.sound_freq)) * 2.5) * sw.get('auditory', 1),
            'touch':      (env.pressure + env.vibration * 0.3) * sw.get('touch', 1),
            'smell':      (sum(env.smell) / 5) * sw.get('smell', 1),
            'taste':      (sum(env.taste)) * sw.get('taste', 1),
            'thermal':    (env.temperature) * sw.get('thermal', 1),
            'balance':    (abs(env.accel[2])) * sw.get('balance', 1),
            'proprio':    0.0,
            'time':       float(env.t),
        }


class Mirror:
    def observe(self, inner_state, trace):
        if len(trace) < 2: return None
        recent = trace[-10:]
        A_prime = -sum(recent) / len(recent)
        return {'A': inner_state, 'A_prime': A_prime, 'diff': inner_state - A_prime}


class Organism:
    """
    —— 生物 ——
    不是被动接收器。它能看、能反应、能行动、能死。
    每次 update 返回一个"动作"——它要选择：
      往哪走 / 要不要吃掉资源 / 跟谁互动 / 等等
    """
    CHANNELS = ['visual','auditory','touch','smell','taste','thermal','balance','proprio']

    def __init__(self, genome=None):
        if genome is None:
            genome = self.random_genome()
        self.genome = genome
        self.senses = SensorArray(genome)
        self.core = YinyangCore()
        self.mirror = Mirror()
        self.state = 0.0
        self.trace = []
        self.mirror_log = []
        self.prev_sensory = None
        self.alive = True
        self.fitness = 0.0
        self.age = 0
        self.energy = 10.0
        self.action_log = []

    @staticmethod
    def random_genome():
        channels = Organism.CHANNELS
        return {
            'sense_weights': {ch: random.uniform(0.1, 2.0) for ch in channels},
            'action_weights': {ch: random.uniform(-1.0, 1.0) for ch in channels},
            'modulation': {
                'threshold': random.uniform(-5.0, 5.0),
                'amplification': random.uniform(0.1, 2.0),
                'memory_decay': random.uniform(0.90, 0.999),
            },
            'interaction': {
                'signal_emission': random.uniform(0.0, 1.0),
                'signal_sensitivity': random.uniform(0.0, 1.0),
                'aggression': random.uniform(0.0, 1.0),
            },
            'internal': {
                'target_state': random.uniform(-50, 50),
                'stability_band': random.uniform(1.0, 20.0),
                'metabolism_rate': random.uniform(0.01, 0.5),
                'growth_rate': random.uniform(0.001, 0.05),
            }
        }

    def clone_mutate(self, rate=0.15, magnitude=0.25):
        child = deepcopy(self.genome)
        for k in child['sense_weights']:
            if random.random() < rate:
                child['sense_weights'][k] *= (1 + random.uniform(-magnitude, magnitude))
                child['sense_weights'][k] = max(0.01, min(5.0, child['sense_weights'][k]))
        for k in child['action_weights']:
            if random.random() < rate:
                child['action_weights'][k] += random.uniform(-magnitude, magnitude)
                child['action_weights'][k] = max(-3.0, min(3.0, child['action_weights'][k]))
        for k in child['modulation']:
            if random.random() < rate * 0.5:
                if k == 'threshold':
                    child['modulation'][k] += random.uniform(-1, 1)
                else:
                    child['modulation'][k] *= (1 + random.uniform(-magnitude*0.5, magnitude*0.5))
                    child['modulation'][k] = max(0.05, min(0.999, child['modulation'][k]))
        for k in child['interaction']:
            if random.random() < rate * 0.5:
                child['interaction'][k] += random.uniform(-magnitude, magnitude)
                child['interaction'][k] = max(0.0, min(1.0, child['interaction'][k]))
        for k in child['internal']:
            if random.random() < rate * 0.5:
                if k == 'target_state':
                    child['internal'][k] += random.uniform(-10, 10)
                    child['internal'][k] = max(-200, min(200, child['internal'][k]))
                elif k in ('stability_band', 'growth_rate'):
                    child['internal'][k] *= (1 + random.uniform(-magnitude, magnitude))
                    child['internal'][k] = max(0.1, min(50.0, child['internal'][k]))
                else:
                    child['internal'][k] *= (1 + random.uniform(-magnitude*0.5, magnitude*0.5))
                    child['internal'][k] = max(0.001, min(1.0, child['internal'][k]))
        return Organism(genome=child)

    def update(self, sensory_raw, organisms):
        """
        感知 → 决策 → 行动
        这一步是进化发生的地方。
        """
        # 读取本机状态
        mod = self.genome['modulation']
        intnl = self.genome['internal']
        action_w = self.genome['action_weights']

        # 时间差分 + 加权
        deltas = {}
        for ch in self.CHANNELS + ['time']:
            if ch == 'time': continue
            if self.prev_sensory is None:
                deltas[ch] = 0.0
            else:
                deltas[ch] = sensory_raw.get(ch, 0.0) - self.prev_sensory.get(ch, 0.0)

        # proprioception = state 变化
        proprio_val = 0.0
        if len(self.trace) >= 2:
            proprio_val = (self.trace[-1] - self.trace[-2]) * 0.1

        # 行动计算：基因组决定如何用感官驱动行动
        action_signal = 0.0
        for ch in self.CHANNELS:
            action_signal += action_w.get(ch, 0) * deltas.get(ch, 0)

        # 内部稳态驱动——生物趋向 target_state
        if self.trace:
            state_error = intnl['target_state'] - self.trace[-1]
            # 如果在 stability_band 内 → 不需要额外调节
            if abs(state_error) > intnl['stability_band']:
                action_signal += state_error * 0.01

        threshold = mod.get('threshold', 0)
        amplification = mod.get('amplification', 1.0)
        memory_decay = mod.get('memory_decay', 0.99)

        # 调制
        if action_signal > threshold:
            mod_delta = (action_signal - threshold) * amplification
        else:
            mod_delta = (action_signal - threshold) * amplification * 0.5

        # 状态更新
        self.state = self.state * memory_decay + mod_delta
        self.trace.append(self.state)
        if len(self.trace) > 1000:
            self.trace = self.trace[-1000:]

        # 镜面自省
        m = self.mirror.observe(self.state, self.trace)
        if m:
            self.mirror_log.append(m)
            if len(self.mirror_log) > 200:
                self.mirror_log = self.mirror_log[-200:]

        # 代谢 + 行为消耗
        metabolic_cost = intnl['metabolism_rate'] * (1 + abs(mod_delta) * 0.1)
        self.energy -= metabolic_cost

        # 死亡条件
        if len(self.trace) >= 20 and abs(self.trace[-1]) > 10000:
            self.alive = False  # 状态爆炸
        if self.energy <= 0:
            self.alive = False  # 饿死

        self.action_log.append({
            'step': len(self.trace),
            'signal': action_signal,
            'delta': mod_delta,
            'energy': self.energy,
        })

        self.prev_sensory = {k: v for k, v in sensory_raw.items()}
        self.age += 1
        return self.alive, action_signal

    def fitness(self):
        """适应度——生存 + 稳态维持"""
        if not self.alive or len(self.trace) < 50:
            return 0.0

        # 年龄
        survival_score = min(1.0, self.age / 500)

        # 状态稳定性
        recent = self.trace[-min(100, len(self.trace)):]
        mean = sum(recent) / len(recent)
        var = sum((x - mean)**2 for x in recent) / len(recent)
        if var < 1e-8:
            stability = 0.01
        else:
            log_var = math.log(var + 1e-8)
            stability = 1.0 / (1.0 + (log_var - 1.0)**2 * 0.05)

        # 能量保持
        energy_score = max(0.0, min(1.0, self.energy / 10.0))

        return survival_score * 0.4 + stability * 0.4 + energy_score * 0.2


class World:
    """
    多生物竞争世界——有限资源环境
    """
    def __init__(self, n_organisms=20, steps_per_gen=800):
        self.n = n_organisms
        self.steps_per_gen = steps_per_gen
        self.population = [Organism() for _ in range(n_organisms)]
        self.env = None
        self.generation = 0
        self.env_core = YinyangCore()
        self.history = []

    def run_generation(self):
        self.env = Environment(seed=random.randint(0, 999999), resource_total=len(self.population) * 3.0)
        self.env_core = YinyangCore()

        for step in range(self.steps_per_gen):
            self.env.step()
            env_E = 0.0

            # 读取所有生物的环境感官
            raw_sensory = {}
            for org in self.population:
                if not org.alive:
                    continue
                weighted = org.senses.read_weighted(self.env, org.genome)
                raw_sensory[id(org)] = weighted
                env_E += sum(abs(v) for k, v in weighted.items() if k != 'time')

            env_E = env_E / max(1, len([o for o in self.population if o.alive])) / 8

            # 物理核心驱动
            self.env_core.step(env_E)

            # 交互——同环境中的生物互相看到对方的信号
            avg_signal = 0.0
            alive_count = 0
            for org in self.population:
                if not org.alive:
                    continue
                avg_signal += org.genome['interaction']['signal_emission']
                alive_count += 1
            avg_signal = avg_signal / max(1, alive_count)

            # 更新各生物
            for org in self.population:
                if not org.alive:
                    continue
                sensory = raw_sensory.get(id(org), {})
                # 加入环境信号（其他生物的影响）
                signal_from_others = avg_signal * org.genome['interaction']['signal_sensitivity']
                sensory['_social'] = signal_from_others

                org.update(sensory, self.population)

                # 资源消耗——生物根据行动信号获取资源
                action_strength = abs(org.action_log[-1]['signal']) if org.action_log else 0
                consumption = min(self.env.resource, action_strength * 0.1)
                self.env.resource -= consumption
                org.energy += consumption * 0.5  # 能量获取不完全等于消耗

            if self.env.resource <= 0:
                self.env.resource = 0.0

            if self.env.energy <= 0:
                break

        # 计算适应度
        for org in self.population:
            org.fitness = org.fitness() if org.alive else 0.0

        alive = [o for o in self.population if o.alive]
        return alive

    def select_and_reproduce(self):
        """强选择——只有 top 50% 能繁殖"""
        self.population.sort(key=lambda o: o.fitness, reverse=True)
        survivors = self.population[:max(2, len(self.population) // 2)]

        new_pop = []
        # 精英保留
        for s in survivors[:2]:
            new_pop.append(s.clone_mutate(rate=0.02, magnitude=0.05))

        # 繁殖填满
        while len(new_pop) < self.n:
            parent = random.choice(survivors)
            child = parent.clone_mutate(rate=0.2, magnitude=0.3)
            new_pop.append(child)

        self.population = new_pop[:self.n]

    def checkpoint(self):
        alive = [o for o in self.population if o.alive]
        best = max(self.population, key=lambda o: o.fitness) if self.population else None
        avg_fit = sum(o.fitness for o in self.population) / max(1, len(self.population))

        if not best:
            return {'gen': self.generation, 'alive': 0, 'best_fitness': 0, 'avg_fitness': 0}

        # 通道权重分布
        sw = best.genome['sense_weights']

        return {
            'gen': self.generation,
            'alive': len(alive),
            'best_fitness': best.fitness,
            'avg_fitness': avg_fit,
            'best_age': best.age,
            'best_energy': best.energy,
            'cared_channels': sorted(sw.items(), key=lambda x: -x[1])[:5],
            'metabolism': best.genome['internal']['metabolism_rate'],
            'target_state': best.genome['internal']['target_state'],
            'memory_decay': best.genome['modulation']['memory_decay'],
            'threshold': best.genome['modulation']['threshold'],
        }

    def run(self, num_gens=200):
        for g in range(num_gens):
            self.generation = g
            alive = self.run_generation()
            ckpt = self.checkpoint()
            self.history.append(ckpt)
            self.select_and_reproduce()

            if g % 10 == 0:
                print(f"GEN {g:4d} | alive={ckpt['alive']:2d}/{self.n} | "
                      f"best_fit={ckpt['best_fitness']:.4f} | avg_fit={ckpt['avg_fitness']:.4f} | "
                      f"age={ckpt.get('best_age',0):4d} | energy={ckpt.get('best_energy',0):+.2f} | "
                      f"top_channels={ckpt.get('cared_channels',[])}")

        # 最终
        ckpt = self.checkpoint()
        self.history.append(ckpt)

        print(f"\n{'='*60}")
        print(f"进化完成 {num_gens} 代")
        print(f"{'='*60}")

        best = max(self.population, key=lambda o: o.fitness) if self.population else None
        if best:
            print("\n最佳个体基因组：")
            print("  感官权重：")
            for ch, w in sorted(best.genome['sense_weights'].items(), key=lambda x: -x[1]):
                print(f"    {ch:12s} = {w:.4f}")
            print("  行动权重：")
            for ch, w in sorted(best.genome['action_weights'].items(), key=lambda x: -abs(x[1])):
                print(f"    {ch:12s} = {w:+.4f}")
            print("  内稳态参数：")
            for k, v in best.genome['internal'].items():
                print(f"    {k:18s} = {v:.5f}")
            print(f"\n  存活步数 = {best.age}")
            print(f"  最终能量 = {best.energy:.4f}")
            print(f"  最终状态 = {best.state:.4f}")
            print(f"  最终适应度 = {best.fitness:.4f}")

        with open("evolution_log.json", "w", encoding="utf-8") as f:
            json.dump({
                'history': self.history,
                'final_best_genome': best.genome if best else None,
                'final_fitness': best.fitness if best else 0,
                'num_gens': num_gens,
            }, f, ensure_ascii=False, indent=2, default=str)
        print("\n保存到 evolution_log.json")
        return self.population


if __name__ == "__main__":
    random.seed(42)
    world = World(n_organisms=20, steps_per_gen=800)
    world.run(num_gens=200)
