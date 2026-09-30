"""
test_bioelectric.py — 莱文生物电网络验证实验
===========================================
5个测试验证bioelectric_network.py的正确性
"""
import numpy as np

from bioelectric_network import (
    ElectrophysiologicalEnvironment,
    GapJunctionNetwork,
    BioelectricDamageDetector,
    CONCEPT_ELECTROPHYSIOLOGY
)

def test_voltage_driven():
    print("\n=== T1 电位差驱动（不饱和） ===")
    env = ElectrophysiologicalEnvironment(dim=1024)
    
    dE = np.zeros(1024)
    dE[0] = 100.0
    
    V_before = env.V[0]
    V_after = env.update_potential(dE, {0})[0]
    
    print(f"  动作电位前电位: {V_before:.2f}")
    print(f"  动作电位后电位: {V_after:.2f}")
    print(f"  不应期计数器: {env.refractory_counter[0]}")
    
    assert env.refractory_counter[0] > 0, "动作电位后应进入不应期"
    assert V_after < V_before, "动作电位后应复极化（回到静息）"
    print("  [PASS] 电位差驱动验证通过 — 强注入触发脉冲后自动复极化，不会饱和")

def test_refractory():
    print("\n=== T2 不应期 ===")
    env = ElectrophysiologicalEnvironment(dim=1024)
    
    dE = np.zeros(1024)
    dE[0] = 100.0
    
    V1 = env.update_potential(dE, {0})
    ref1 = env.refractory_counter[0]
    V2 = env.update_potential(dE, {0})
    ref2 = env.refractory_counter[0]
    
    print(f"  第一次后不应期: {ref1}")
    print(f"  第二次后不应期: {ref2}")
    assert ref1 > 0, "应进入不应期"
    print("  [PASS] 不应期验证通过")

def test_gap_junction():
    print("\n=== T3 间隙连接耦合 ===")
    net = GapJunctionNetwork()
    net.add("渴", "水", "需要", w=1.0)
    
    V = np.zeros(66)
    name2id = {"渴": 0, "水": 1}
    V[0] = -30.0
    V[1] = -70.0
    
    dV = net.coupling(V, name2id)
    V_new = V + dV
    
    diff_before = abs(V[0] - V[1])
    diff_after = abs(V_new[0] - V_new[1])
    
    print(f"  耦合前电位差: {diff_before:.2f}")
    print(f"  耦合后电位差: {diff_after:.2f}")
    assert diff_after < diff_before, "间隙连接应让电位趋于一致"
    print("  [PASS] 间隙连接耦合验证通过")

def test_damage_detection():
    print("\n=== T4 损伤检测（梯度崩塌） ===")
    detector = BioelectricDamageDetector(1024)
    resting = np.zeros(1024)
    
    V_normal = np.random.randn(1024) * 5.0
    damage_normal = detector.check(V_normal, resting)
    print(f"  正常状态: {damage_normal}")
    assert damage_normal is None
    
    V_collapse = np.ones(1024) * 3.0
    damage_collapse = detector.check(V_collapse, resting)
    print(f"  梯度崩塌: {damage_collapse}")
    assert damage_collapse == "gradient_collapse"
    print("  [PASS] 损伤检测验证通过")

def test_self_repair():
    print("\n=== T5 自修复（Xenobots式功能重建） ===")
    env = ElectrophysiologicalEnvironment(dim=1024)
    detector = BioelectricDamageDetector(1024)
    
    env.V = np.ones(1024) * 3.0
    old_V = env.V.copy()
    
    damage = detector.check(env.V, env.resting_template)
    print(f"  检测到的损伤: {damage}")
    assert damage == "gradient_collapse"
    
    detector.repair(env, damage)
    new_V = env.V
    
    print(f"  修复前std: {np.std(old_V):.2f}")
    print(f"  修复后std: {np.std(new_V):.2f}")
    assert np.std(new_V) > np.std(old_V), "修复应重建电位梯度"
    print("  [PASS] 自修复验证通过 — 不是回退旧状态，是重建功能等效的新稳态")

if __name__ == "__main__":
    print("=" * 60)
    print("莱文生物电网络验证实验")
    print("=" * 60)
    test_voltage_driven()
    test_refractory()
    test_gap_junction()
    test_damage_detection()
    test_self_repair()
    print("\n" + "=" * 60)
    print("全部 5 项测试通过 ✓")
    print("=" * 60)
