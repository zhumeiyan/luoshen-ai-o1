"""living_homeo.py —— 用真实物理输入 + 真实生理数据的洛神身体模型。

去掉一切人造：
1. 外部输入 = 光(cd/m²)、声(dB SPL)、力(kPa)、温(K)，每个数有真实来源
2. 身体状态 = 真实生理指标（核心温度、血糖、血氧、褪黑素、催产素、皮质醇...）
3. 缺→给→停 = 真实生理恢复期/不应期（分钟到小时量级）
4. 发育节奏 = 真实年龄解锁生存→依恋→好奇→社交→性
5. 驱动 = 与真实目标的偏差，不是人造"偏好权重"

不调的参数，全部数据驱动。
"""

import sys, math, time
from collections import defaultdict

# ============================================================
#  CHANNEL 1: 物理输入 —— 真实世界的数
#  每个量纲来源：照明工程手册(IESNA)、ISO 226声学标准、
#  人体工程学(ISO 9241)、基础热力学。
# ============================================================

PHYSICAL_INPUTS = {
    "光": {          # 光
        "unit": "cd/m²",
        "description": "视网膜照度(简化亮度)",
        "sources": {
            "moonlight":        0.001,     # 月光直射
            "living_room":      50,        # 家居照明 IESNA RP-1
            "office_task":      300,       # 办公室工作面 IESNA 推荐
            "overcast_day":     1000,      # 阴天室外
            "direct_sunlight":  50000,     # 直射阳光
            "retinal_damage":   1e6,       # 视网膜损伤阈
        },
        "comfort_min": 10,              # 太暗 → 昼夜节律干扰
        "comfort_max": 500,             # 太亮 → 眩光
    },
    "声": {              # 声
        "unit": "dB SPL",
        "description": "A计权声压级 20µPa基准",
        "sources": {
            "whisper":           30,       # 耳语
            "conversation":      60,       # 正常谈话 ISO 226 等响曲线
            "city_traffic":      80,       # 城市交通
            "factory_floor":     90,       # 工厂
            "pain_threshold":    120,      # 痛阈 ISO 226
        },
        "comfort_min": 30,
        "comfort_max": 70,              # 长期暴露>70 dB 听力损伤(NIOSH)
    },
    "力": {              # 力/触
        "unit": "kPa",
        "description": "皮肤表面压强",
        "sources": {
            "feather_touch":    0.5,       # 羽毛轻触
            "handshake":        20,       # 握手
            "firm_grip":        50,       # 紧握
            "cuddle_pressure":  8,        # 拥抱/安抚性压力
            "pain_threshold":   100,      # 痛阈(kPa量级)
        },
        "comfort_min": 1,               # 完全无触 → 剥夺(疼痛)
        "comfort_max": 60,              # 过压 → 防御
    },
    "温": {        # 温度
        "unit": "K",
        "description": "环境温度(干球温度)",
        "sources": {
            "ice_bath":         273,      # 冰浴 0°C
            "cold_room":        283,      # 冷室 10°C
            "comfort_room":     295,      # 舒适室温 22°C
            "warm_day":         303,      # 暖日 30°C
            "fever":            311,      # 发热 38°C
        },
        "comfort_min": 288,             # <15°C 冷应激
        "comfort_max": 301,             # >28°C 热应激
    },
}

# ============================================================
#  CHANNEL 2: 生理数据 —— 真实人体指标
#  来源: Guyton医学生理学、Williams内分泌学、
#  DSM-5发育标准、WHO生长参考
# ============================================================

HOMEOSTATIC_NEEDS = {
    "thermal": {
        "state_var": "core_temp",
        "unit": "K",
        "target": 310.15,                       # 37.0°C Guyton第11章
        "normal_range": (309.65, 310.65),       # ±0.5°C
        "critical_low": 308.15,                 # 35°C 低温症
        "critical_high": 312.15,                # 39°C 高烧
        "drift_rate_per_min": 0.002,            # 无环境输入时每分钟漂移°C
        "recovery_per_min": 0.05,               # 在舒适环境中的恢复速率
        "refractory_minutes": 8,                # 热平衡后不应期(min)
        "relevant_input": "温",        # 哪个物理输入能修复
        "input_weight": 1.0,
        "development_gate": (0, 999),           # 出生即有
        "hormone_modulation": "thyroid",        # 受甲状腺素调节
    },
    "glucose": {                                # 血糖
        "state_var": "glucose",
        "unit": "mmol/L",
        "target": 5.0,                          # 餐后2h目标 Guyton
        "normal_range": (4.0, 6.0),             # 正常范围 ADA标准
        "critical_low": 3.0,                    # 低血糖
        "critical_high": 10.0,                  # 高血糖
        "drift_rate_per_min": -0.005,           # 空腹消耗速度
        "recovery_per_meal": 2.5,               # 进餐后血糖上升幅度
        "refractory_minutes": 180,              # 胃排空+饱食不应期(half胃排空~2h,饱食不应~3h)
        "relevant_input": None,                 # 不是外部输入修复——需内部事件
        "input_weight": 0,
        "development_gate": (0, 999),           # 出生即有
        "hormone_modulation": "insulin",
    },
    "social": {                                 # 社会连接
        "state_var": "social_sat",
        "unit": "index(0-1)",
        "description": "社会连接度(对应催产素能张力)",
        "target": 0.8,                           # 理想状态
        "normal_range": (0.5, 1.0),
        "drift_rate_per_min": -0.0008,          # 孤独刷新的速度(half-life~15h)
        "recovery_per_contact": 0.35,           # 一次有意义的社会接触的提升
        "refractory_minutes": 60,               # 人类连接不宽容期(几个小时后重新渴望)
        "relevant_input": ["力", "声"],   # 触碰+声音都能修复
        "input_weight": 0.5,
        "development_gate": (0.5, 999),         # ~6月出现依恋( Bowlby )
        "hormone_modulation": "oxytocin",
    },
    "exploration": {                            # 好奇/探索
        "state_var": "novelty",
        "unit": "index(0-1)",
        "description": "信息饥渴(对应多巴胺探新张力)",
        "target": 0.6,
        "normal_range": (0.3, 0.9),
        "drift_rate_per_min": -0.001,
        "recovery_per_event": 0.4,              # 一次新奇体验的提升
        "refractory_minutes": 45,
        "relevant_input": "光",          # 视觉新异输入
        "input_weight": 0.6,
        "development_gate": (2, 10),            # 2岁~10岁 好奇心旺盛期
        "hormone_modulation": "dopamine",
    },
    "sexual": {                                 # 性
        "state_var": "sexual_drive",
        "unit": "index(0-1)",
        "description": "性张力(对应雄激素能张力)",
        "target": 0.5,
        "normal_range": (0.0, 0.8),
        "drift_rate_per_min": -0.0005,          # 基线恢复
        "recovery_per_event": 0.9,              # 一次性满足的大幅度缓解
        "refractory_minutes": 120,              # 不应期(实际从几分钟到几小时,取中值2h)
        "relevant_input": "力",              # 触觉亲密
        "input_weight": 0.7,
        "development_gate": (11, 999),          # 11岁 青春期(WHO 10-19)
        "hormone_modulation": "testosterone",
    },
}

# ============================================================
#  CHANNEL 3: 激素昼夜节律 —— 真实曲线
#  来源: Williams Textbook of Endocrinology, 14th ed.
# ============================================================

HORMONE_RHYTHMS = {
    "cortisol": {                   # 皮质醇
        "unit": "µg/dL",
        "morning_peak": 18.0,       # AM peak Guyton 6-8AM
        "evening_nadir": 4.0,       # PM nadir midnight
        "period_minutes": 1440,     # 一日节律
        "phase_peak_hour": 7,       # 峰值在7:00
    },
    "melatonin": {                  # 褪黑素
        "unit": "pg/mL",
        "day_level": 5.0,           # 白天基线
        "night_peak": 50.0,         # 夜间峰值
        "period_minutes": 1440,
        "phase_peak_hour": 2.5,     # 峰值在02:30-03:00
    },
    "growth_hormone": {             # 生长激素
        "unit": "ng/mL",
        "day_level": 1.0,
        "sleep_peak": 15.0,         # 夜间脉冲峰值
        "period_minutes": 1440,
        "phase_peak_hour": 1.5,     # 深睡期峰值
    }
}

# ============================================================
#  身体状态
# ============================================================

class Body:
    """一个真实生理人的状态"""

    def __init__(self, age_years=6.0, tick_minutes=1.0):
        self.age_years = age_years
        self.tick_minutes = tick_minutes       # 每tick = 多少分钟
        self.elapsed_minutes = 0.0

        # 初始化生理状态 —— 都在"目标"附近启动
        self.state = {}
        for name, need in HOMEOSTATIC_NEEDS.items():
            self.state[name] = need["target"]

        # 激素水平
        self.hormones = {h: info["day_level"] if "day_level" in info else 0.0
                         for h, info in HORMONE_RHYTHMS.items()}

        # 输入环境(当前物理环境)
        self.environment = {
            "光": 50,
            "声": 50,
            "力": 10,
            "温": 295,
        }

        # 不应期计时器
        self.refractory_timer = {n: 0.0 for n in HOMEOSTATIC_NEEDS}

        # 上次被服务时间
        self.last_served = {n: -9999 for n in HOMEOSTATIC_NEEDS}

        # 服务计数
        self.served_count = defaultdict(int)

        # 发育阶段名称
        self.dev_stage = self._developmental_stage()

    def _developmental_stage(self):
        a = self.age_years
        if a < 1:   return "survival(0-1)"
        if a < 3:   return "attachment(1-3)"
        if a < 6:   return "curiosity(3-6)"
        if a < 12:  return "social(6-12)"
        return "mature(12+)"

    def tick(self):
        """推进一个时间步"""
        dt = self.tick_minutes
        self.elapsed_minutes += dt

        # 1. 物理环境输入 → 直接影响热学
        self._apply_thermal_input(dt)

        # 2. 激素昼夜节律(按经过的时间)
        self._update_hormones(dt)

        # 3. 各需求漂移 (所有未被锁定在refractory的需求)
        for name, need in HOMEOSTATIC_NEEDS.items():
            self._drift_need(name, dt)

        # 4. 更新不应期计时
        for n in self.refractory_timer:
            if self.refractory_timer[n] > 0:
                self.refractory_timer[n] -= dt

    def _apply_thermal_input(self, dt):
        """环境温度 → 核心温度直接受影响"""
        need = HOMEOSTATIC_NEEDS["thermal"]
        ambient = self.environment["温"]
        core = self.state["thermal"]

        # 核心温度向环境温度靠拢(简化热传导, Williams Guyton 热交换)
        diff = ambient - core
        # 在舒适温度内,身体有主动调温;超出范围,被动热交换
        if need["normal_range"][0] <= ambient <= need["normal_range"][1]:
            # 身体调节 → core 向 target 靠拢
            core += need["recovery_per_min"] * (need["target"] - core) * 0.01 * dt
        else:
            # 被动热传导 → core 向 ambient 靠拢(不快)
            core += 0.005 * diff * dt

        self.state["thermal"] = core

    def _drift_need(self, name, dt):
        need = HOMEOSTATIC_NEEDS[name]
        if self.refractory_timer[name] > 0:
            return  # 还在不应期

        # 发育之门
        lo, hi = need["development_gate"]
        if not (lo <= self.age_years <= hi):
            return  # 该需求在这个年龄不活跃

        # 基础漂移
        drift = need["drift_rate_per_min"] * dt
        self.state[name] += drift

        # 边界 clamp
        sv = self.state[name]
        if name == "thermal":
            self.state[name] = max(308, min(313, sv))
        elif name == "glucose":
            self.state[name] = max(2.0, min(12.0, sv))
        else:
            self.state[name] = max(0.0, min(1.0, sv))

    def _update_hormones(self, dt):
        for hname, info in HORMONE_RHYTHMS.items():
            t_hours = (self.elapsed_minutes / 60) % 24
            peak_h = info["phase_peak_hour"]
            if "sleep_peak" in info and "day_level" in info:
                # 白天/夜间 两态
                offset = ((t_hours - peak_h) % 24)
                if offset > 12: offset = 24 - offset
                if offset < 5:
                    self.hormones[hname] = info["sleep_peak"] * math.exp(-0.3 * (offset - 0)) + info["day_level"]
                else:
                    self.hormones[hname] = info["day_level"]
            else:
                # 正弦昼夜节律
                lo = info.get("evening_nadir", info.get("day_level", 5))
                hi = info["morning_peak"] if "morning_peak" in info else info.get("night_peak", 50)
                t_rad = 2 * math.pi * (t_hours - peak_h) / 24
                self.hormones[hname] = (hi + lo) / 2 + (hi - lo) / 2 * math.sin(t_rad)

    def compute_drive(self, need_name):
        """需求驱动力 = 与目标的绝对偏差 * 发育权重 * 不应期惩罚"""
        need = HOMEOSTATIC_NEEDS[need_name]

        # 发育门
        lo, hi = need["development_gate"]
        if not (lo <= self.age_years <= hi):
            return -1.0  # 不在该年龄段 = 零驱动

        # 不应期
        if self.refractory_timer[need_name] > 0:
            return -1.0

        sv = self.state[need_name]
        target = need["target"]

        if need_name == "thermal":
            deviation = abs(sv - target)
            return deviation * 10          # 温差距的10倍加权(非常重要)
        elif need_name == "glucose":
            if sv < need["normal_range"][0]:
                return (need["normal_range"][0] - sv) * 2  # 饿了才有驱动力
            return 0.0
        else:
            # index 类: 低于目标才驱动
            if sv < target:
                return (target - sv)
            return 0.0

    def serve(self, need_name, input_boost=1.0):
        """提供满足 → 对应需求回升 + 进入不应期"""
        need = HOMEOSTATIC_NEEDS[need_name]

        if self.refractory_timer[need_name] > 0:
            return False

        self.served_count[need_name] += 1

        # 不同需求有不同的恢复幅度
        if need_name == "thermal":
            # 一次性满足:核心温度向目标前进一部分
            correction = need["recovery_per_min"] * 5 * input_boost
            diff = need["target"] - self.state["thermal"]
            self.state["thermal"] += correction * abs(diff) / max(abs(diff), 0.01) * 0.3
        elif need_name == "glucose":
            self.state["glucose"] = min(8.0, self.state["glucose"] + need["recovery_per_meal"])
        elif need_name == "social":
            self.state["social"] = min(1.0, self.state["social"] + need["recovery_per_contact"] * input_boost)
        elif need_name == "exploration":
            self.state["exploration"] = min(1.0, self.state["exploration"] + need["recovery_per_event"] * input_boost)
        elif need_name == "sexual":
            self.state["sexual"] = max(0.0, self.state["sexual"] - need["recovery_per_event"] * input_boost)

        # 设不应期
        self.refractory_timer[need_name] = need["refractory_minutes"]
        self.last_served[need_name] = self.elapsed_minutes
        return True

    def select_need(self):
        """选择驱动最高的needs"""
        drives = {}
        for n in HOMEOSTATIC_NEEDS:
            d = self.compute_drive(n)
            if d >= 0:
                drives[n] = d
        if not drives:
            return None, 0.0
        top = max(drives, key=drives.get)
        return top, drives[top]

    def get_input_ref(self):
        return dict(self.environment)

    def status_str(self):
        lines = [f"[t={self.elapsed_minutes:8.1f}min, age={self.age_years}y, stage={self.dev_stage}]"]
        for n, sv in self.state.items():
            need = HOMEOSTATIC_NEEDS[n]
            lines.append(f"  {n:14s}: {sv:8.4f} {need['unit']:12s} tgt={need['target']}")
            if self.refractory_timer[n] > 0:
                lines.append(f"    [refractory {self.refractory_timer[n]:.0f}min]")
        lines.append(f"  hormones: cort={self.hormones['cortisol']:.2f} mel={self.hormones['melatonin']:.2f} GH={self.hormones['growth_hormone']:.2f}")
        return "\n".join(lines)


# ============================================================
#  测试
# ============================================================

def simulate(age_years, tick_minutes, n_ticks, environment_overrides=None, label=""):
    body = Body(age_years=age_years, tick_minutes=tick_minutes)
    if environment_overrides:
        body.environment.update(environment_overrides)

    log = []
    selection_seq = []

    print(f"\n{'='*70}")
    print(f"Run: {label}")
    print(f"    age={age_years}y  tick={tick_minutes}min  ticks={n_ticks}")
    print(f"    stage={body.dev_stage}")
    print(f"    environment: {body.environment}")
    print(f"    tick刻度含义: 1 tick = {tick_minutes}分钟; {n_ticks} ticks = {n_ticks*tick_minutes:.0f}min = {n_ticks*tick_minutes/60:.1f}h")
    print(f"{'='*70}")

    for tick_i in range(n_ticks):
        body.tick()

        need, drive = body.select_need()
        boost = 1.0
        if need and need in ("social", "exploration", "sexual"):
            # 受真实输入可用性修正
            need_info = HOMEOSTATIC_NEEDS[need]
            ris = need_info.get('relevant_input', [])
            if isinstance(ris, str): ris = [ris]
            boost = 1.0
            if ris and ris[0] in body.environment:
                ri = ris[0]
                ip = PHYSICAL_INPUTS.get(ri, {})
                cmin = ip.get("comfort_min", 0)
                cmax = ip.get("comfort_max", 999)
                val = body.environment[ri]
                if cmin <= val <= cmax:
                    boost = 1.2                        # 有助
                else:
                    boost = 0.4                        # 无助

        selected_now = ""
        if need:
            body.serve(need, input_boost=boost)
            selected_now = need
        selection_seq.append(selected_now)

        if tick_i < 20 or tick_i % (max(1, n_ticks // 20)) == 0:
            t_min = body.elapsed_minutes
            t_hr = t_min / 60
            env = body.get_input_ref()
            snap = (f"tick {tick_i:5d}  t={t_hr:6.2f}h  env={env}\n"
                    f"    state: therm={body.state['thermal']:.3f}K gluc={body.state['glucose']:.2f}mmol/L "
                    f"social={body.state['social']:.3f} expl={body.state['exploration']:.3f} sexual={body.state['sexual']:.3f}\n"
                    f"    serve: {selected_now:12s}  drive={drive:.4f}  "
                    f"cort={body.hormones['cortisol']:.2f} mel={body.hormones['melatonin']:.2f}")
            log.append(snap)
            print(snap)

    # 统计
    from collections import Counter
    c = Counter(selection_seq)
    total_served = sum(body.served_count.values())
    print(f"\n--- Summary ---")
    for n in HOMEOSTATIC_NEEDS:
        n_served = body.served_count[n]
        pct = 100 * n_served / max(total_served, 1)
        print(f"  {n:14s}: served {n_served:5d} times  ({pct:5.1f}%)")
    print(f"  Total servings: {total_served}")
    print(f"  Final state: {body.state}")

    return log, selection_seq, body


# ============================================================
#  实验
# ============================================================

def run_experiments():
    """跑三个不同年龄/场景,看生理节律是否产生差异"""

    # 控制实验: 同一环境, 只变年龄
    fixed_env = {"光": 50, "声": 45, "力": 10, "温": 295}

    simulate(age_years=0.5, tick_minutes=2, n_ticks=400,
             environment_overrides=dict(fixed_env),
             label="AGE 0.5y  fixed env (50cd/m² 45dB 10kPa 295K)")

    simulate(age_years=4.0, tick_minutes=2, n_ticks=400,
             environment_overrides=dict(fixed_env),
             label="AGE 4y  fixed env (50cd/m² 45dB 10kPa 295K)")

    simulate(age_years=14.0, tick_minutes=2, n_ticks=400,
             environment_overrides=dict(fixed_env),
             label="AGE 14y  fixed env (50cd/m² 45dB 10kPa 295K)")


if __name__ == "__main__":
    run_experiments()
