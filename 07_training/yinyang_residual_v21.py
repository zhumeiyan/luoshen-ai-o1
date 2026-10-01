#!/usr/bin/env python3
"""
yinyang_residual_v21 — 反推协议
========================================
对v21最佳个体进行残差分析（反推）：
  1. 逐区域残差画像（哪类信号难预测？）
  2. 误差动态（收敛/发散/震荡？）
  3. 器官活性分析（饱和？死亡？过度激活？）
  4. 跨通道耦合（通道间关联强度？）
  5. 突现统计（猜想：涌现的时刻在哪里？）

反推之后，这些发现将用于正推（修改架构）。
"""
import random, math, copy, json, sys
sys.path.insert(0, '.')
from yinyang_brain_v21_fullorg import Pool, WM, Org, N_CHANNELS, N_REGIONS, DELAY

def load_best_genome():
    """从v21_run.txt无法直接读取，重新跑一遍v21找到最佳个体"""
    # 实际使用时从独立genome文件读。这里先用演化引擎输出
    pass

class ResidualProtocol:
    """残差反推协议"""
    
    def __init__(self, org):
        self.org = org
        self.g = org.g
        self.wm = org.wm
        self.n = org.g['n']
    
    def run_full_eval(self, ticks=600, seed=42, switch_interval=30):
        """
        带回放的反推评估：
        记录每个tick的完整状态，供事后分析。
        """
        pk = Pool(seed=seed); pk.reset()
        self.wm.reset()
        cr = random.randint(0, 8)
        
        # 记录序列
        records = []
        
        # 暖机
        for _ in range(30):
            s = pk.sense(cr)
            self.wm.fwd(s, update=True)
            pk.step()
        
        # 正式评估
        for t in range(ticks):
            s = pk.sense(cr)
            pred = self.wm.fwd(s, update=True)
            pk.step()
            ns = pk.sense(cr)
            pk.step()
            
            mse = sum((ns[j]-pred[j])**2 for j in range(N_CHANNELS)) / N_CHANNELS
            self.wm.learn(pred, ns)
            
            # 记录完整状态
            rec = {
                't': t,
                'mse': mse,
                'channel_errs': [ns[j]-pred[j] for j in range(N_CHANNELS)],
                'pred': pred[:],
                'actual': ns[:],
                'h_sample': self.wm.h[:self.n],
                'surprise': self.wm.surprise_level,
                'sens_adapt': self.wm.sens_adapt[:],
                'homeo': dict(self.wm.homeo),
                'replay_buf_size': len(self.wm.replay_buffer),
                'spike_count': self.wm.organ_activity['spike_count'],
                'energy_use': self.wm.organ_activity['energy_use'],
                'saturation': self.wm.organ_activity['saturation'],
                'region': cr,
            }
            records.append(rec)
            
            if t % switch_interval == switch_interval - 1:
                cr = random.randint(0, 8)
        
        return records
    
    # ==== L1: 区域残差画像 ====
    def analyze_regional_fingerprint(self, records):
        """按环境区域分组，统计残差特征"""
        buckets = {i: [] for i in range(N_REGIONS)}
        for rec in records:
            buckets[rec['region']].append(rec['mse'])
        
        results = {}
        for rid in range(N_REGIONS):
            vals = buckets[rid]
            if not vals:
                results[rid] = {'count': 0, 'mean_mse': -1}
                continue
            mean_mse = sum(vals) / len(vals)
            sorted_vals = sorted(vals)
            p50 = sorted_vals[len(sorted_vals)//2]
            p90 = sorted_vals[int(len(sorted_vals)*0.9)]
            results[rid] = {
                'count': len(vals),
                'mean_mse': round(mean_mse, 5),
                'p50_mse': round(p50, 5),
                'p90_mse': round(p90, 5),
                'type': Pool(seed=42).r[rid]['type'],
                'sigma': Pool(seed=42).r[rid]['sigma'],
            }
        return results
    
    # ==== L2: 误差动态 ====
    def analyze_error_dynamics(self, records):
        """误差时序分析：趋势、自相关、收敛"""
        mses = [r['mse'] for r in records]
        n = len(mses)
        
        # 分段统计
        seg_size = 50
        segments = []
        for i in range(0, n, seg_size):
            seg = mses[i:i+seg_size]
            segments.append(sum(seg)/len(seg))
        
        # 趋势（最后5段vs前5段）
        head = sum(segments[:min(5,len(segments))]) / min(5, len(segments))
        tail = sum(segments[-min(5,len(segments)):]) / min(5, len(segments))
        trend = 'improving' if tail < head * 0.9 else ('worsening' if tail > head * 1.1 else 'stable')
        
        # 自相关（lag 1）
        if n > 1:
            mean_mse = sum(mses) / n
            autocov = sum((mses[i]-mean_mse)*(mses[i+1]-mean_mse) for i in range(n-1)) / (n-1)
            var = sum((m-mean_mse)**2 for m in mses) / n
            autocorr = autocov / var if var > 0 else 0
        else:
            autocorr = 0
        
        # 收敛速率（指数拟合）
        # 计算移动平均的衰减
        if len(segments) > 5:
            delta = [abs(segments[i+1]-segments[i]) for i in range(len(segments)-1)]
            avg_delta = sum(delta[-10:]) / min(10, len(delta))
        else:
            avg_delta = -1
        
        return {
            'overall_mean_mse': round(sum(mses)/n, 5),
            'trend': trend,
            'head_5seg_mse': round(head, 5),
            'tail_5seg_mse': round(tail, 5),
            'autocorr_lag1': round(autocorr, 4),
            'avg_segment_delta': round(avg_delta, 5),
            'segment_mse': [round(s, 4) for s in segments],
        }
    
    # ==== L3: 器官活性分析 ====
    def analyze_organ_activity(self, records):
        """全器官活性/饱和/死亡分析"""
        n = self.n
        
        # 隐藏状态统计
        h_mat = [r['h_sample'] for r in records]
        
        death_count = 0  # |h| < 0.01
        sat_count = 0   # |h| > 0.98
        active_count = 0  # 0.01 <= |h| <= 0.98
        total = 0
        for h in h_mat:
            for v in h:
                total += 1
                if abs(v) < 0.01: death_count += 1
                elif abs(v) > 0.98: sat_count += 1
                else: active_count += 1
        
        # 感觉适应统计
        adapt_records = [r['sens_adapt'] for r in records[-100:]]
        avg_adapt = [sum(adapt_records[t][c] for t in range(len(adapt_records))) / len(adapt_records) for c in range(N_CHANNELS)]
        
        # 惊奇动态
        surp_records = [r['surprise'] for r in records]
        avg_surprise = sum(surp_records) / len(surp_records)
        max_surprise = max(surp_records)
        
        # 下丘脑驱力变化
        homeo_records = [r['homeo'] for r in records]
        avg_homeo = {k: sum(r[k] for r in homeo_records)/len(homeo_records) for k in ['hunger','curiosity','coherence','engagement']}
        
        # 海马体回放频率
        replay_sizes = [r['replay_buf_size'] for r in records]
        avg_replay = sum(replay_sizes) / len(replay_sizes)
        
        # 脉冲/能量/饱和
        spikes = [r['spike_count'] for r in records]
        energy = [r['energy_use'] for r in records]
        sat_r = [r['saturation'] for r in records]
        
        return {
            'n_nodes': n,
            'death_ratio': round(death_count/max(1,total), 4),
            'saturation_ratio': round(sat_count/max(1,total), 4),
            'active_ratio': round(active_count/max(1,total), 4),
            'sens_adapt_avg': [round(x,4) for x in avg_adapt],
            'surprise_avg': round(avg_surprise, 4),
            'surprise_max': round(max_surprise, 4),
            'homeo_avg': {k: round(v,3) for k,v in avg_homeo.items()},
            'replay_buf_avg': round(avg_replay, 2),
            'spike_count_avg': round(sum(spikes)/len(spikes), 2),
            'energy_use_avg': round(sum(energy)/len(energy), 4),
            'saturation_series_avg': round(sum(sat_r)/len(sat_r), 4),
        }
    
    # ==== L4: 跨通道耦合 ====
    def analyze_cross_channel(self, records):
        """通道间残差协方差矩阵"""
        n = len(records)
        # 计算每个通道的残差序列
        seqs = {c: [] for c in range(N_CHANNELS)}
        for rec in records:
            for c in range(N_CHANNELS):
                seqs[c].append(rec['channel_errs'][c])
        
        # 均值
        means = {c: sum(seqs[c])/len(seqs[c]) for c in range(N_CHANNELS)}
        
        # 协方差
        cov = {}
        for ci in range(N_CHANNELS):
            for cj in range(N_CHANNELS):
                s = sum((seqs[ci][k]-means[ci])*(seqs[cj][k]-means[cj]) for k in range(n)) / max(1,n-1)
                cov[(ci,cj)] = round(s, 6)
        
        # 找出最强耦合对
        top_pairs = sorted([(k,v) for k,v in cov.items() if k[0]!=k[1]], key=lambda x: -abs(x[1]))[:5]
        
        # 找出方差最大的通道（废差大＝难预测）
        var_per_ch = {c: sum((x-means[c])**2 for x in seqs[c])/max(1,n-1) for c in range(N_CHANNELS)}
        worst_ch = max(var_per_ch, key=var_per_ch.get)
        best_ch = min(var_per_ch, key=var_per_ch.get)
        
        return {
            'variance_per_channel': {k: round(v,5) for k,v in var_per_ch.items()},
            'worst_channel': (worst_ch, round(var_per_ch[worst_ch],5)),
            'best_channel': (best_ch, round(var_per_ch[best_ch],5)),
            'top_coupled_pairs': [({'ch': k, 'name': f'{k[0]}-{k[1]}'}, v) for k,v in top_pairs],
            'full_cov_matrix_diagonal': [round(cov[(c,c)],6) for c in range(N_CHANNELS)],
        }
    
    # ==== L5: 突变/涌现检测 ====
    def analyze_emergence_moments(self, records):
        """寻找涌现时刻：误差突然改善 / 突然恶化"""
        mses = [r['mse'] for r in records]
        window = 20
        events = []
        
        for i in range(window, len(mses)-window):
            before = sum(mses[i-window:i]) / window
            after = sum(mses[i:i+window]) / window
            if before > 0.01:
                change = (after - before) / before
                if abs(change) > 0.5:  # 50%以上变化
                    events.append({
                        'tick': i,
                        'change': round(change, 3),
                        'before_mse': round(before, 5),
                        'after_mse': round(after, 5),
                        'surprise_at': round(records[i]['surprise'], 4),
                        'region': records[i]['region'],
                    })
        
        # 重大事件排序
        events.sort(key=lambda e: -abs(e['change']))
        
        # 找涌现时刻（大幅改善）
        improvements = [e for e in events if e['change'] < -0.5]
        degradations = [e for e in events if e['change'] > 0.5]
        
        return {
            'total_change_events': len(events),
            'improvement_events': len(improvements),
            'degradation_events': len(degradations),
            'top_improvements': improvements[:10],
            'top_degradations': degradations[:10],
        }
    
    def run_all(self):
        """运行完整反推协议"""
        print("[1/5] 运行600tick评估...")
        records = self.run_full_eval(ticks=600)
        
        print("[2/5] L1 区域残差画像...")
        L1 = self.analyze_regional_fingerprint(records)
        
        print("[3/5] L2 误差动态...")
        L2 = self.analyze_error_dynamics(records)
        
        print("[4/5] L3 器官活性...")
        L3 = self.analyze_organ_activity(records)
        
        print("[5/5] L4 跨通道耦合...")
        L4 = self.analyze_cross_channel(records)
        
        print("[6/6] L5 涌现分析...")
        L5 = self.analyze_emergence_moments(records)
        
        return {
            'genome_summary': {
                'n': self.n,
                'lr': self.g['lr'],
                'fold_lr': self.g['fold_lr'],
                'sens_adapt_rate': self.g['sens_adapt_rate'],
                'hip_lr': self.g['hip_lr'],
                'hip_replay_n': self.g['hip_replay_n'],
                'amy_thr': self.g['amy_thr'],
                'pfc_gate_thr': self.g['pfc_gate_thr'],
                'pfc_gate_lr': self.g['pfc_gate_lr'],
                'bg_explore_rate': self.g['bg_explore_rate'],
                'bg_noise': self.g['bg_noise'],
                'fold_threshold': self.g['fold_threshold'],
                'sens_gain': [round(x,3) for x in self.g['sens_gain']],
                'assoc_w': [round(x,3) for x in self.g['assoc_w']],
                'sparse_thr': round(self.g['sparse_thr'],3),
                'hypo_drive': [round(x,3) for x in self.g['hypo_drive']],
                'hypo_fill': round(self.g['hypo_fill'],4),
                'leak_avg': round(sum(self.g['leak'])/len(self.g['leak']),4),
            },
            'L1_regional': L1,
            'L2_dynamics': L2,
            'L3_organ_activity': L3,
            'L4_cross_channel': L4,
            'L5_emergence': L5,
        }


def run_on_v21_best():
    """对v21最佳个体运行反推"""
    print("=" * 60)
    print("yinyang_residual_v21 — 反推协议")
    print("=" * 60)
    
    # 重新演化v21，记录最佳个体
    from yinyang_brain_v21_fullorg import Engine
    
    print("\n演化v21获取最佳个体...")
    eng = Engine(n_org=20, n_gen=300, seed=42, eval_ticks=350)
    eng.evolve()
    
    best_org = Org(copy.deepcopy(eng.best_g))
    print(f"\n最佳个体: n={eng.best_g['n']}, fitness={eng.best_ever:.4f}")
    
    # 运行反推
    proto = ResidualProtocol(best_org)
    results = proto.run_all()
    
    # 输出报告
    print("\n" + "=" * 60)
    print("反推报告")
    print("=" * 60)
    
    gs = results['genome_summary']
    print(f"\n[基因组] n={gs['n']} lr={gs['lr']:.4f} fold_lr={gs['fold_lr']:.4f}")
    print(f" 感觉: gain={gs['sens_gain']}, adapt={gs['sens_adapt_rate']:.3f}")
    print(f" 联合: w={gs['assoc_w']}, sparse_thr={gs['sparse_thr']}")
    print(f" 海马: hip_lr={gs['hip_lr']}, replay_n={gs['hip_replay_n']}")
    print(f" 杏仁: thr={gs['amy_thr']}")
    print(f" 前额: gate_thr={gs['pfc_gate_thr']}, gate_lr={gs['pfc_gate_lr']}")
    print(f" 基节: explore={gs['bg_explore_rate']}, noise={gs['bg_noise']}")
    print(f" 下丘: drive={gs['hypo_drive']}, fill={gs['hypo_fill']}")
    print(f" 小脑: fold_threshold={gs['fold_threshold']}")
    
    L1 = results['L1_regional']
    print(f"\n[L1 区域残差画像]")
    for rid in range(N_REGIONS):
        r = L1[rid]
        if r['count'] > 0:
            print(f"  region {rid} ({r['type']:>8s} σ={r['sigma']:.2f}): mean_mse={r['mean_mse']:.5f} p50={r['p50_mse']:.5f} p90={r['p90_mse']:.5f} (n={r['count']})")
    
    L2 = results['L2_dynamics']
    print(f"\n[L2 误差动态]")
    print(f"  整体MSE: {L2['overall_mean_mse']:.5f}")
    print(f"  趋势: {L2['trend']}")
    print(f"  前段: {L2['head_5seg_mse']:.5f} → 后段: {L2['tail_5seg_mse']:.5f}")
    print(f"  自相关(lag1): {L2['autocorr_lag1']}")
    print(f"  均段变化: {L2['avg_segment_delta']}")
    print(f"  各段MSE: {L2['segment_mse']}")
    
    L3 = results['L3_organ_activity']
    print(f"\n[L3 器官活性]")
    print(f"  死亡比: {L3['death_ratio']:.4f}")
    print(f"  饱和比: {L3['saturation_ratio']:.4f}")
    print(f"  活跃比: {L3['active_ratio']:.4f}")
    print(f"  感觉适应(avg): {L3['sens_adapt_avg']}")
    print(f"  惊奇(avg={L3['surprise_avg']}, max={L3['surprise_max']})")
    print(f"  下丘脑驱力: {L3['homeo_avg']}")
    print(f"  回放缓冲(avg): {L3['replay_buf_avg']}")
    print(f"  脉冲(avg/tick): {L3['spike_count_avg']}")
    print(f"  能量消耗: {L3['energy_use_avg']}")
    print(f"  饱和序列: {L3['saturation_series_avg']}")
    
    L4 = results['L4_cross_channel']
    print(f"\n[L4 跨通道耦合]")
    print(f"  通道方差: {L4['variance_per_channel']}")
    print(f"  最难预测: ch{L4['worst_channel'][0]} (var={L4['worst_channel'][1]:.5f})")
    print(f"  最易预测: ch{L4['best_channel'][0]} (var={L4['best_channel'][1]:.5f})")
    print(f"  最强耦合: {L4['top_coupled_pairs'][:3]}")
    
    # L5简化输出
    L5 = results['L5_emergence']
    print(f"\n[L5 涌现事件]")
    print(f"  总突变: {L5['total_change_events']}")
    print(f"  改善事件: {L5['improvement_events']}")
    print(f"  恶化事件: {L5['degradation_events']}")
    if L5['top_improvements']:
        print(f"  最大改善时刻:")
        for e in L5['top_improvements'][:5]:
            print(f"    tick={e['tick']} chg={e['change']:+.1%} before={e['before_mse']:.4f}→after={e['after_mse']:.4f} surp={e['surprise_at']:.3f}")
    
    # 保存
    with open('v21_residual_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n完整报告已保存: v21_residual_report.json")
    
    return results


if __name__ == '__main__':
    run_on_v21_best()
