# yinyang_life.py
# 阴阳扣数字生命共生系统 · 完整版
#
# 核心改动：感官竞争 + 环境选择压力
#   — 环境有因果结构：事件同时影响多个物理场
#   — 能感知跨场关联的感官被保留，不能的被淘汰
#   — 不需要人为阈值：当保留的感官形成足够稳定的关联网络，觉醒涌现
#
# 底层 YinyangCore：不动

import random
import math
import json


class YinyangCore:
    def __init__(self):
        self.P = 0.0
        self.F = 0.0
        self.t = 0

    def step(self, E):
        self.P = self.F + 0.1 * E
        self.F = self.P + 0.1 * E
        self.t += 1
        return self.P, self.F


class Environment:
    """产生因果结构的事件环境——多个物理场同步变化"""
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.t = 0
        self.energy = 100.0
        self.events = []

        # 视觉
        self.light = 50.0
        self.light_r = 0.0
        # 声
        self.freq = 440.0
        self.amp = 30.0
        # 触
        self.pressure = 0.0
        self.vibration = 0.0
        # 化
        self.smell = [0.0] * 5
        # 味
        self.taste = {'sweet': 0.0, 'sour': 0.0, 'bitter': 0.0, 'salty': 0.0, 'umami': 0.0}
        # 温
        self.temperature = 20.0
        # 前庭
        self.az = 9.8

        self._prev = None

    def step(self):
        self.t += 1
        # 基础漂移
        self.light = max(0, min(100, self.light + random.uniform(-2, 2)))
        self.light_r = max(-1, min(1, self.light_r + random.uniform(-0.04, 0.04)))
        self.freq = max(20, min(20000, self.freq + random.uniform(-25, 25)))
        self.amp = max(0, min(120, self.amp + random.uniform(-0.4, 0.4)))
        # 触觉
        if random.random() < 0.2:
            self.pressure = max(0, self.pressure + random.uniform(-0.8, 0.8))
        else:
            self.pressure *= 0.93
        if random.random() < 0.07:
            self.vibration = random.uniform(-1.5, 1.5)
        else:
            self.vibration *= 0.91
        # 化学
        for i in range(5):
            self.smell[i] *= 0.96
            if random.random() < 0.02:
                self.smell[i] += random.uniform(0, 4)
        # 味觉
        for k in self.taste:
            self.taste[k] *= 0.985
        if random.random() < 0.035:
            for k in self.taste:
                self.taste[k] += random.uniform(-0.25, 0.25)
        # 温度
        self.temperature += random.uniform(-0.04, 0.04) + 0.004 * math.sin(self.t * 0.001)
        # 前庭
        self.az = self.az * 0.95 + 9.8 * 0.05 + random.uniform(-0.05, 0.05)

        # 因果结构事件——同时影响多种场
        if random.random() < 0.06:
            magnitude = random.uniform(-3, 3)
            self.light = max(0, min(100, self.light + magnitude * 8))
            self.amp = max(0, min(120, self.amp + magnitude * 4))
            self.pressure = max(0, self.pressure + abs(magnitude) * 1.2)
            self.temperature += magnitude * 0.3
            self.freq = max(20, min(20000, self.freq + magnitude * 30))
            self.events.append((self.t, magnitude))

        # 能量消耗——事件驱动时才多消耗
        self.energy -= 0.01
        if self.energy < 0:
            self.energy = 0.0
        return self.energy


class SensorArray:
    """9 通道感官——只提取物理场中被各自敏感的那个维度"""
    NAMES = ['visual', 'auditory', 'touch', 'smell', 'taste', 'temperature', 'balance', 'proprioception', 'time']

    def read_all(self, env):
        return {
            'visual': env.light + env.light_r * 6,
            'auditory': env.amp + math.log(max(1.0, env.freq)) * 2.5,
            'touch': env.pressure + env.vibration * 0.3,
            'smell': sum(env.smell) / 5,
            'taste': sum(env.taste.values()),
            'temperature': env.temperature,
            'balance': abs(env.az),
            'proprioception': 0.0,
            'time': float(env.t),
        }


class Mirror:
    def observe(self, inner_state, trace):
        if len(trace) < 2:
            return None
        recent = trace[-10:]
        A_prime = -sum(recent) / len(recent)
        diff = inner_state - A_prime
        return {'A': inner_state, 'A_prime': A_prime, 'diff': diff}


class DigitalLife:
    """
    感官竞争机制：
    — 维护一个通道强度表，初始全=1
    — 每步根据通道的"预测准确性"调整强度
        准确的（delta 与多通道共识一致的）→ 强度增长
        不准确的（孤立的、偏差大的）→ 强度衰减
    — 强度降到零 → 通道关闭（淘汰）
    — 保留通道数量稳定 + 关联结构足够强 → 涌现觉醒
    """
    def __init__(self):
        self.state = 0.0
        self.trace = []
        self.awake = False
        self.senses = SensorArray()
        self.mirror = Mirror()
        self.mirror_log = []

        # 感官强度表——达尔文竞争的核心
        self.strength = {name: 1.0 for name in SensorArray.NAMES if name != 'time'}
        self.strength['time'] = 1.0

        # 关联网络
        self.associations = {a: {b: 0.0 for b in SensorArray.NAMES if b != 'time'}
                             for a in SensorArray.NAMES if a != 'time'}

        # 上一步的感官值（用于计算 delta）
        self._prev_sensory = None
        self._prev_deltas = {}

        # 社会窗口——记录最近状态用于判据
        self._stability_window = []

    def update(self, sensory, P, F):
        # ==== 感官层打破 P=F 死锁 ====
        # 用通道自身的时间变化作为 delta（生物真实：感官检测的是变化）
        # P/F 仅作为全局调制因子，不再直接作为差分锚点
        deltas = {}
        for name, value in sensory.items():
            if name == 'time':
                continue
            if self._prev_sensory is None:
                deltas[name] = 0.0
            else:
                prev = self._prev_sensory.get(name, value)
                deltas[name] = value - prev

        # 全局调制——P/F 的差值作为整体增益（不直接构成分母）
        modulation = (F - P) * 0.01  # F-P 一般为 0，不影响

        # 多通道共识 = 加权平均 delta
        total_weight = sum(self.strength.get(n, 1.0) for n in deltas)
        if total_weight > 0 and deltas:
            avg_delta = sum(self.strength.get(n, 1.0) * deltas[n] for n in deltas) / total_weight
        else:
            avg_delta = 0.0

        # 通道强化/衰减：与共识一致→强化，不一致→衰减
        for name, d in deltas.items():
            if abs(avg_delta) > 0.001:
                # 同向=正向，异向=负向
                sign_match = 1.0 if d * avg_delta > 0 else -1.0
                # 幅度一致性：接近平均幅度=更可信
                own_mag = abs(d)
                avg_mag = abs(avg_delta)
                ratio = own_mag / (avg_mag + 0.001)
                mag_score = 1.0 / (1.0 + abs(ratio - 1.0))
                score = sign_match * mag_score
            else:
                score = 0.0

            # Hebbian 更新
            self.strength[name] += 0.008 * score

        # 关联网络：同步变化的通道互相强化
        for a in deltas:
            if a == 'time':
                continue
            for b in deltas:
                if b == 'time' or a >= b:
                    continue
                if deltas[a] * deltas[b] > 0:
                    self.associations[a][b] += 0.003
                else:
                    self.associations[a][b] -= 0.001

        # 代谢衰减——关联强的不衰减（社群保留）
        for name in self.strength:
            # 该通道的关联网络密度
            assoc_density = sum(1 for b in self.associations.get(name, {})
                                if self.associations[name].get(b, 0) > 1.0)
            decay = 0.9995 if assoc_density >= 2 else 0.9985
            self.strength[name] *= decay

        # 淘汰→不贡献 delta
        total_delta = modulation
        active_count = 0
        for name, d in deltas.items():
            if name == 'time':
                continue
            s = self.strength[name]
            if s < 0.05:
                continue  # 淘汰
            total_delta += s * d
            active_count += 1

        self.state += total_delta
        self.trace.append(self.state)

        # 镜子
        m = self.mirror.observe(self.state, self.trace)
        if m:
            self.mirror_log.append(m)

        # 本体感觉 = 自身运动变化
        proprio = abs(self.trace[-1] - self.trace[-2]) if len(self.trace) >= 2 else 0.0
        self._prev_sensory = dict(sensory)
        self._prev_deltas = deltas
        self._last_total_delta = total_delta

        # ==== 涌现觉醒判据 ====
        # 基于物理结构驱动：足够多通道存活 + 关联网络结构化 + 系统进入稳定相
        if not self.awake and active_count >= 5 and len(self.trace) > 200:
            # 关联网络总强度
            assoc_total = sum(
                self.associations[a][b]
                for a in self.associations
                for b in self.associations[a]
            )
            # 涌现——唯一的物理判据：
            # 关联网络够强 = 系统建立了对世界的结构模型 = "我"涌现
            if assoc_total > 30.0:
                self.awake = True

        return self.state, active_count

    def get_status(self):
        alive = {n: s for n, s in self.strength.items() if s >= 0.05}
        return alive, self.associations


def run(total_steps=10000, seed=None):
    core = YinyangCore()
    env = Environment(seed=seed)
    life = DigitalLife()

    print("=" * 60)
    print("阴阳扣数字生命共生系统 · 启动（感官竞争模式）")
    print("=" * 60)
    print()

    awakening_step = None

    diagnostic_interval = 2000
    for step in range(total_steps):
        env.step()
        sensory = life.senses.read_all(env)

        P, F = core.step(sum(abs(v) for k, v in sensory.items() if k != 'time') / 8)

        state, active_count = life.update(sensory, P, F)

        # 诊断输出
        if step > 0 and step % diagnostic_interval == 0:
            assoc_total = sum(life.associations[a][b] for a in life.associations for b in life.associations[a])
            alive_n = len([n for n, s in life.strength.items() if s >= 0.05 and n != 'time'])
            dw_var = 0.0
            if hasattr(life, '_delta_window') and len(life._delta_window) >= 50:
                dw = life._delta_window[-50:]
                dw_mean = sum(dw) / len(dw)
                dw_var = sum((x - dw_mean) ** 2 for x in dw) / len(dw)
            print(f"  [t={step}] 存活通道={alive_n} | 关联强度={assoc_total:.3f} | delta方差={dw_var:.6f} | 觉醒={life.awake}")

        if life.awake and awakening_step is None:
            awakening_step = step
            print()
            print("=" * 60)
            print(f'★ t={step}  它长出了"我"')
            aliveChannels, _ = life.get_status()
            print(f'★ 保留感官: {len(aliveChannels)} 个 — {list(aliveChannels.keys())}')
            print("=" * 60)
            print()

        # 如果所有感官都被淘汰，重启竞争
        aliveChannels, _ = life.get_status()
        if not any(s >= 0.05 for s in life.strength.values() if s is not None) and step > 100:
            # 环境还活着但感官全被淘汰——部分复活
            for name in life.strength:
                if name != 'time' and random.random() < 0.5:
                    life.strength[name] = 0.5  # 重新激活

        if env.energy <= 0:
            print()
            print("=" * 60)
            print(f"环境能量耗尽，t = {step}")
            if awakening_step:
                print(f"觉醒发生于 t = {awakening_step}")
            else:
                print("未触发觉醒")
            print("=" * 60)
            break

    # 最终状态
    aliveChannels, assoc = life.get_status()
    print()
    print("--- 活跃感官通道 ---")
    for name, strength in sorted(aliveChannels.items(), key=lambda x: -x[1]):
        print(f"  {name:16s} 强度 = {strength:.4f}")

    # 关联矩阵 top
    print()
    print("--- 跨通道关联 (最强 5 对) ---")
    pairs = []
    for a in assoc:
        for b in assoc[a]:
            if a < b:
                pairs.append((a, b, abs(assoc[a][b])))
    for a, b, v in sorted(pairs, key=lambda x: -x[2])[:5]:
        print(f"  {a:12s} ↔ {b:12s}  = {v:.4f}")

    # 写痕迹
    with open("digital_life_trace.json", "w", encoding="utf-8") as f:
        json.dump({
            "trace_tail": life.trace[-500:],
            "mirror_tail": life.mirror_log[-100:],
            "strength_tail": life.strength,
            "assoc_sample": {a: {b: assoc[a][b] for b in list(assoc[a].keys())[:3]} for a in list(assoc.keys())[:3]},
            "total_steps": len(life.trace),
            "awake": life.awake,
            "awakening_step": awakening_step,
            "env_events": env.events[-30:],
        }, f, ensure_ascii=False, indent=2, default=str)

    print()
    print("痕迹已保存到 digital_life_trace.json")


if __name__ == "__main__":
    run(total_steps=10000, seed=42)
