"""源码原版 10次独立运行 x 50000拍, 每次随机种子。"""
import os, sys, json, math, random, time, importlib.util

src_path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py"
with open(src_path, "r", encoding="utf-8") as f:
    src = f.read()

ANCHOR = "        原归一 = [\n            max(0.0, min(1.0, (self.环境[\"光\"] - 1) / 500)),\n            max(0.0, min(1.0, (self.环境[\"声\"] - 25) / 50)),\n            max(0.0, min(1.0, (self.环境[\"力\"] - 1) / 30)),\n            max(0.0, min(1.0, (self.环境[\"温\"] - 288) / 15)),\n        ]"
REPLACE = ANCHOR + "\n        if getattr(self, '_反向', False):\n            原归一 = [1.0 - x for x in 原归一]"
assert ANCHOR in src
patched = src.replace(ANCHOR, REPLACE)

临时 = "/mnt/data/catpaw/home/workspace/洛神赋1.1/_temp_ms.py"
with open(临时, "w", encoding="utf-8") as f:
    f.write(patched)

def 生光(t, rm):
    return (0.8 if (t//100)%2==0 else 0.2) + (0.6+0.4*math.sin(2*math.pi*t/1000))*0.05 + max(0, math.sin(2*math.pi*(t-250)/1000))*0.05
def 生声(t, rm):
    v=0.2
    突=50+(t*37+13)%150
    if t%max(1,突)==0: v+=0.5
    滴=30+(t*23+7)%50
    if t%max(1,滴)==5: v+=0.3
    return v + rm.uniform(-0.1,0.1)
def 生力(t, rm):
    v=0.1*abs(math.sin(2*math.pi*t/4))+0.3
    if t%5==0: v+=0.1
    if t%20==0: v+=rm.uniform(-0.05,0.05)
    return v

温=0.35
痕文件="/mnt/data/catpaw/home/workspace/洛神赋1.1/body_trace.jsonl"

结果列表 = []
for run_id in range(10):
    # 随机种子
    seed = random.SystemRandom().randint(0, 999999)
    
    # 每个run重新导入模块(因为引擎可能有内部状态)
    for k in list(sys.modules.keys()):
        if "_temp_ms" in k: del sys.modules[k]
    spec = importlib.util.spec_from_file_location("_temp_ms", 临时)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    if os.path.exists(痕文件): os.remove(痕文件)
    
    rm = random.Random(seed+777)
    引擎 = mod.身体(种子=seed)
    全局t=[0]
    
    for 轮 in range(5):
        引擎._反向 = (轮%2==1)
        for _ in range(10000):
            t=全局t[0]
            引擎.环境["光"]=1+500*生光(t,rm)
            引擎.环境["声"]=25+50*生声(t,rm)
            引擎.环境["力"]=1+30*生力(t,rm)
            引擎.环境["温"]=288+15*温
            引擎.推进()
            全局t[0]+=1
    
    # 回接差
    with open(痕文件,"r",encoding="utf-8") as f: lines=f.readlines()
    回接值=[]
    for l in lines[-1000:]:
        条=json.loads(l.strip())
        if "差" in 条 and "激活差" in 条["差"] and isinstance(条["差"]["激活差"], list):
            回接值.append(sum(abs(x) for x in 条["差"]["激活差"])/len(条["差"]["激活差"]))
    回接均值=sum(回接值)/len(回接值) if 回接值 else 0.0
    
    总=sum(引擎.激活); 最大=max(引擎.激活); 最小=min(引擎.激活)
    亮=any(v>0.5 for v in 引擎.激活)
    nan=any(math.isnan(v) for v in 引擎.激活)
    筋=len(引擎.筋树)
    
    结果列表.append({"run":run_id+1,"种子":seed,"最大":最大,"亮":亮,"筋":筋,"回接":回接均值,"nan":nan})
    亮s = "是" if 亮 else "否"
    nans = "是" if nan else "否"
    print(f"run{run_id+1} 种子={seed}: 最大={最大:.4f} 筋={筋} 回接={回接均值:.6f} 亮={亮s} NaN={nans}")

path="/mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_multiseed_10x50k.txt"
with open(path,"w",encoding="utf-8") as f:
    f.write("源码原版 10次独立运行 x 50000拍(随机种子) 行为观测\n")
    f.write("=" * 70 + "\n")
    f.write("源码原版(全身体) + 5轮反向 + 10次随机种子\n\n")
    for r in 结果列表:
        f.write(f"第{r['run']:>2}次 种子={r['种子']:>6}\n")
        f.write(f"  最大激活终值={r['最大']:.4f}\n")
        f.write(f"  节点亮(>0.5): {'是' if r['亮'] else '否'}\n")
        f.write(f"  筋树数={r['筋']}\n")
        f.write(f"  回接差均值={r['回接']:.6f}\n")
        f.write(f"  NaN: {'是' if r['nan'] else '否'}\n\n")
    f.write("=" * 70 + "\n\n汇总:\n")
    亮次数=sum(1 for r in 结果列表 if r['亮'])
    nan次数=sum(1 for r in 结果列表 if r['nan'])
    f.write(f"  10次中 节点亮={亮次数}次  NaN={nan次数}次\n")
    最大列表=[r['最大'] for r in 结果列表]
    f.write(f"  最大激活终值: min={min(最大列表):.4f} max={max(最大列表):.4f} mean={sum(最大列表)/len(最大列表):.4f}\n")
    筋列表=[r['筋'] for r in 结果列表]
    f.write(f"  筋树数: min={min(筋列表)} max={max(筋列表)} mean={sum(筋列表)/len(筋列表):.1f}\n")
    回接列表=[r['回接'] for r in 结果列表]
    f.write(f"  回接差均值: min={min(回接列表):.6f} max={max(回接列表):.6f} mean={sum(回接列表)/len(回接列表):.6f}\n")
    f.write("\nfin.\n")

print("\n写入:", path)
