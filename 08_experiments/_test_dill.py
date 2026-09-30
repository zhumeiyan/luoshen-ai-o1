import dill, sys, math, random
src_path = "/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py"
with open(src_path, encoding="utf-8") as f:
    src = f.read()
ns = {}
exec(compile(src, src_path, "exec"), ns)
Body = ns["身体"]
eng = Body(种子=7)
for _ in range(1000):
    eng.环境["光"] = 500; eng.环境["声"] = 50; eng.环境["力"] = 5; eng.环境["温"] = 300
    eng.推进()
print(f"before: t={eng.时刻} sum={sum(eng.激活):.4f}")
# dill roundtrip
data = dill.dumps(eng)
print(f"dill size={len(data)}")
# Save and load (simulating process restart)
with open("/tmp/test_eng.pkl", "wb") as f:
    f.write(data)
with open("/tmp/test_eng.pkl", "rb") as f:
    eng2 = dill.load(f)
print(f"after:  t={eng2.时刻} sum={sum(eng2.激活):.4f}")
# 继续推进两者
for _ in range(1000):
    eng.环境["光"] = 500; eng.环境["声"] = 50; eng.环境["力"] = 5; eng.环境["温"] = 300
    eng.推进()
    eng2.环境["光"] = 500; eng2.环境["声"] = 50; eng2.环境["力"] = 5; eng2.环境["温"] = 300
    eng2.推进()
match = all(abs(eng.激活[i] - eng2.激活[i]) < 1e-12 for i in range(len(eng.激活)))
print(f"match after 1000 more: {match}")
