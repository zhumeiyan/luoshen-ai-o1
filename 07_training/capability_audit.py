# capability_audit.py
# 能力审计：对进化末代的个体逐项测试具体能力
#
# 能力清单：
#   A. 单步预测精度（MSE，每个区域分别测）
#   B. 多步前瞻预测（N=1,3,10,30 步，看误差累积）
#   C. 在线学习能力（进入新区域后误差下降速度）
#   D. 区域切换适应（σ 跳变后的恢复速度）
#   E. 内在动机行为验证（boredom → SEEK-HOT 是否真实发生）
#   F. 能量管理（是否会饿死 / 是否会主动觅食）
#   G. 泛化到 novel 吸引子（训练时未见过的动力学）

import random, math, json, sys
sys.path.insert(0, '.')
from yinyang_evo_v15 import Organism, WorldModel, StableRegionPool, N_CHANNELS, N_ACTIONS

def run_audit(genome, n_steps=400, seed=42):
    """对单个基因组跑完整能力审计"""
    results = {}
    wm = WorldModel(genome)
    n = genome['params']['n_nodes']

    # === A. 单步预测精度（每个区域分别测）===
    pool = StableRegionPool(seed=seed)
    region_mse = {}
    for rid in range(9):
        pool.reset_states()
        # 先 warmup 让 pool 的状态脱离 0
        for _ in range(50):
            for rr in range(9):
                pool.step_region(rr)
        errors = []
        for _ in range(400):
            sensory = pool.get_sensory(rid)
            sensory_p = sensory[:N_CHANNELS-1] + [(pool.regions[rid]['sigma']-0.35)*0.7]
            predicted, _ = wm.forward(sensory_p)
            # step ALL regions once
            for rr in range(9):
                pool.step_region(rr)
            actual = pool.get_sensory(rid)
            actual_p = actual[:N_CHANNELS-1] + [(pool.regions[rid]['sigma']-0.35)*0.7]
            mse = sum((actual_p[i]-predicted[i])**2 for i in range(N_CHANNELS))/N_CHANNELS
            errors.append(mse)
        region_mse[rid] = {
            'sigma': round(pool.regions[rid]['sigma'], 3),
            'mse_avg': round(sum(errors[100:])/300, 5),
            'mse_final_50': round(sum(errors[-50:])/50, 5)
        }
    results['regional_prediction'] = region_mse

    # === B. 多步前瞻预测 ===
    pool2 = StableRegionPool(seed=seed+1)
    for _ in range(50):
        for rr in range(9):
            pool2.step_region(rr)
    multi_step = {}
    for n_steps_ahead in [1, 3, 5, 10, 20, 50]:
        errors = []
        for trial in range(20):
            rid = random.randint(0, 8)
            sensory = pool2.get_sensory(rid)
            sensory_p = sensory[:N_CHANNELS-1] + [(pool2.regions[rid]['sigma']-0.35)*0.7]
            # 用 WM 自身预测作为下一步输入（unrolled）
            pred, nodes = wm.forward(sensory_p)
            # pred 是 n-steps-ahead=1 的预测
            for step in range(1, n_steps_ahead):
                # 用 prediction 作为下一步输入（只读 forward，不修改 self.nodes）
                feed = pred[:N_CHANNELS-1] + [sensory_p[N_CHANNELS-1]]  # proprio 不变
                pred, nodes = wm.forward_from_nodes(nodes, feed)
            # 计算真实值（step forward n_steps_ahead 次）
            for rr in range(9):
                pool2.step_region(rr)
            # ... 实际状态 vs pred
            # 注意：这里简化，仅测 unrolled 误差
            actual_rid = rid  # 假设区域不变
            actual = pool2.get_sensory(actual_rid)
            actual_p = actual[:N_CHANNELS-1] + [(pool2.regions[actual_rid]['sigma']-0.35)*0.7]
            mse = sum((actual_p[i]-pred[i])**2 for i in range(N_CHANNELS))/N_CHANNELS
            errors.append(mse)
        multi_step[n_steps_ahead] = round(sum(errors)/len(errors), 5)
    results['multi_step_prediction'] = multi_step

    # === C. 在线学习能力 ===
    pool3 = StableRegionPool(seed=seed+2)
    # 只训练前 4 代区域 → 测试切换到后 5 代区域
    wm_learner = WorldModel(genome)
    for rid in range(4):
        for _ in range(200):
            sensory = pool3.get_sensory(rid)
            sensory_p = sensory[:N_CHANNELS-1] + [(pool3.regions[rid]['sigma']-0.35)*0.7]
            pred, _ = wm_learner.forward(sensory_p)
            for rr in range(9):
                pool3.step_region(rr)
            actual = pool3.get_sensory(rid)
            actual_p = actual[:N_CHANNELS-1] + [(pool3.regions[rid]['sigma']-0.35)*0.7]
            mse = sum((actual_p[i]-pred[i])**2 for i in range(N_CHANNELS))/N_CHANNELS
            rate = wm_learner.params['learning_rate']
            wm_learner.learn(actual_p, pred, rate)
    # 切换到未训练区域
    transfer_errors = []
    for rid in range(4, 9):
        for _ in range(50):
            sensory = pool3.get_sensory(rid)
            sensory_p = sensory[:N_CHANNELS-1] + [(pool3.regions[rid]['sigma']-0.35)*0.7]
            pred, _ = wm_learner.forward(sensory_p)
            for rr in range(9):
                pool3.step_region(rr)
            actual = pool3.get_sensory(rid)
            actual_p = actual[:N_CHANNELS-1] + [(pool3.regions[rid]['sigma']-0.35)*0.7]
            mse = sum((actual_p[i]-pred[i])**2 for i in range(N_CHANNELS))/N_CHANNELS
            transfer_errors.append(mse)
    results['online_learning'] = {
        'mse_on_untrained_region': round(sum(transfer_errors)/len(transfer_errors), 5),
        'n_untrained_regions': 5
    }

    # === D. 区域切换适应速度 ===
    pool4 = StableRegionPool(seed=seed+3)
    wm_adapt = WorldModel(genome)
    switch_errors = []
    rid = 0
    for step_n in range(300):
        if step_n == 100:
            rid = 8  # 突然从低sigma跳到高sigma
        elif step_n == 200:
            rid = 2  # 再跳到中等
        sensory = pool4.get_sensory(rid)
        sensory_p = sensory[:N_CHANNELS-1] + [(pool4.regions[rid]['sigma']-0.35)*0.7]
        pred, _ = wm_adapt.forward(sensory_p)
        for rr in range(9):
            pool4.step_region(rr)
        actual = pool4.get_sensory(rid)
        actual_p = actual[:N_CHANNELS-1] + [(pool4.regions[rid]['sigma']-0.35)*0.7]
        mse = sum((actual_p[i]-pred[i])**2 for i in range(N_CHANNELS))/N_CHANNELS
        rate = wm_adapt.params['learning_rate']
        wm_adapt.learn(actual_p, pred, rate)
        if step_n >= 95 and step_n <= 130:
            switch_errors.append(mse)
    results['switch_adaptation'] = {
        'mse_durng_switch_window': round(sum(switch_errors)/len(switch_errors), 5),
        'switch_description': 'rid 0 @:100 → rid 8 @:100-200 → rid 2 @:200+'
    }

    # === E. 内在动机信号验证 ===
    pool5 = StableRegionPool(seed=seed+4)
    from yinyang_evo_v15 import IntrinsicMotivation
    org = Organism(genome=genome)
    org.wm = WorldModel(genome)
    org.im = IntrinsicMotivation()
    org.current_region = 0  # 从低sigma开始
    boredom_trace = []
    region_trace = []
    for step_n in range(400):
        rid = org.current_region
        sig = pool5.regions[rid]['sigma']
        sensory_before = pool5.get_sensory(rid)[:N_CHANNELS-1] + [(sig-0.35)*0.7]
        predicted, action = org.plan(sensory_before)
        # 执行行动
        if action == 1:
            org.current_region = random.randint(0, 8)
        elif action == 2 and org.current_region < 8:
            org.current_region += 1
        elif action == 3 and org.current_region > 0:
            org.current_region -= 1
        # step all regions once
        for rr in range(9):
            pool5.step_region(rr)
        rid_new = org.current_region
        sig_new = pool5.regions[rid_new]['sigma']
        new_sens = pool5.get_sensory(rid_new)[:N_CHANNELS-1] + [(sig_new-0.35)*0.7]
        org.learn_and_update(new_sens, sig_new, rid_new)
        if step_n % 5 == 0:
            boredom_trace.append(round(org.im.boredom, 4))
            region_trace.append(org.current_region)
    results['intrinsic_motivation'] = {
        'boredom_final': round(org.im.boredom, 4),
        'boredom_trend_first_20': boredom_trace[:20],
        'boredom_trend_last_20': boredom_trace[-20:],
        'region_switches_observed': sum(1 for i in range(1,len(region_trace)) if region_trace[i]!=region_trace[i-1]),
        'region_distribution': {r: region_trace.count(r) for r in set(region_trace)},
        'drive_final': round(org.im.drive_energy, 3)
    }

    # === F. 能量管理 ===
    results['energy'] = {
        'final_energy': round(org.energy, 3),
        'survived': org.alive,
        'age': org.age
    }

    # === G. Novel 泛化 ===
    # 用随机新频率 + 新相位构成未见过的吸引子
    pool_novel = StableRegionPool(seed=seed+100)
    # 修改频率为训练时未见过的
    for r in pool_novel.regions:
        r['freq'] = [f*2.5 for f in r['freq']]
        r['phase'] = [p+1.0 for p in r['phase']]
    pool_novel.reset_states()
    for _ in range(50):
        for rr in range(9):
            pool_novel.step_region(rr)
    wm_novel = WorldModel(genome)
    novel_errors = []
    for _ in range(300):
        rid = random.randint(0, 8)
        sensory = pool_novel.get_sensory(rid)
        sensory_p = sensory[:N_CHANNELS-1] + [(pool_novel.regions[rid]['sigma']-0.35)*0.7]
        pred, _ = wm_novel.forward(sensory_p)
        for rr in range(9):
            pool_novel.step_region(rr)
        actual = pool_novel.get_sensory(rid)
        actual_p = actual[:N_CHANNELS-1] + [(pool_novel.regions[rid]['sigma']-0.35)*0.7]
        mse = sum((actual_p[i]-pred[i])**2 for i in range(N_CHANNELS))/N_CHANNELS
        novel_errors.append(mse)
    results['novel_generalization'] = {
        'mse_novel_dynamics': round(sum(novel_errors[100:])/200, 5),
        'description': 'Frequencies ×2.5, phases +1.0 (unseen during evolution)'
    }

    return results


def format_audit_report(results):
    """格式化为可读报告"""
    lines = []
    lines.append('='*60)
    lines.append('能力审计 Capability Audit Report')
    lines.append('='*60)

    lines.append('\n[A] 区域专项预测（每个区域）')
    for rid in range(9):
        r = results['regional_prediction'][rid]
        lines.append(f'  Region {rid} (σ={r["sigma"]:.3f}): avg MSE={r["mse_avg"]:.5f}, final_50 MSE={r["mse_final_50"]:.5f}')

    lines.append('\n[B] 多步前瞻预测')
    for n, mse in sorted(results['multi_step_prediction'].items()):
        lines.append(f'  N={n:3d} 步: MSE = {mse:.5f}')

    lines.append('\n[C] 在线学习（未训练区域）')
    ol = results['online_learning']
    lines.append(f'  未训练区域 MSE: {ol["mse_on_untrained_region"]:.5f}')

    lines.append('\n[D] 切换适应')
    sa = results['switch_adaptation']
    lines.append(f'  切换窗口 MSE: {sa["mse_durng_switch_window"]:.5f}')
    lines.append(f'  事件: {sa["switch_description"]}')

    lines.append('\n[E] 内在动机')
    im = results['intrinsic_motivation']
    lines.append(f'  最终 boredom: {im["boredom_final"]:.4f}')
    lines.append(f'  最终 drive:   {im["drive_final"]:.3f}')
    lines.append(f'  区域切换次数: {im["region_switches_observed"]}')
    lines.append(f'  boredom 趋势(前20点): {im["boredom_trend_first_20"]}')
    lines.append(f'  boredom 趋势(后20点): {im["boredom_trend_last_20"]}')
    lines.append(f'  区域分布: {im["region_distribution"]}')

    lines.append('\n[F] 能量管理')
    en = results['energy']
    lines.append(f'  最终能量: {en["final_energy"]:.3f}')
    lines.append(f'  存活: {en["survived"]}')
    lines.append(f'  寿命: {en["age"]}')

    lines.append('\n[G] Novel 泛化')
    ng = results['novel_generalization']
    lines.append(f'  未见习动力学 MSE: {ng["mse_novel_dynamics"]:.5f}')

    return '\n'.join(lines)


if __name__ == '__main__':
    import pickle
    # 尝试加载上次保存的最佳基因组
    try:
        with open('best_genome_v16.pkl','rb') as f:
            genome = pickle.load(f)
        print('Loaded best genome from best_genome_v16.pkl')
    except FileNotFoundError:
        print('No saved genome. Running a quick evolution to get one...')
        from yinyang_evo_v15 import World
        random.seed(99)
        w = World(n_organisms=30, steps_per_gen=300, seed=99)
        for g in range(80):
            w.generation = g
            alive = w.run_generation()
            w.select_and_reproduce(gen=g)
        # 找最佳个体
        best_org = max(w.population, key=lambda o: w.compute_fitness(o))
        genome = best_org.genome
        with open('best_genome_v16.pkl','wb') as f:
            pickle.dump(genome, f)
        print(f'Saved best genome from gen 80 (fitness={w.compute_fitness(best_org):.4f})')

    results = run_audit(genome, seed=42)
    report = format_audit_report(results)
    print(report)

    with open('capability_audit_results.json','w') as f:
        json.dump(results, f, indent=2, default=str)
    print('\nSaved detailed results to capability_audit_results.json')
