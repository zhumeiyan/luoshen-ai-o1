import pickle, sys
src_path = '/mnt/data/catpaw/home/workspace/洛神赋1.1/身体稳态.py'
with open(src_path, encoding='utf-8') as f:
    src = f.read()
ns = {}
exec(compile(src, src_path, 'exec'), ns)
Body = ns['身体']
eng = Body(种子=7)
# 跑100步
for _ in range(100):
    eng.环境["光"] = 500; eng.环境["声"] = 50; eng.环境["力"] = 5; eng.环境["温"] = 300
    eng.推进()
data = pickle.dumps(eng)
print(f"PICKLE_OK size={len(data)}")
eng2 = pickle.loads(data)
print(f"UNPICKLE_OK nodes={len(eng2.激活)} t={eng2.时刻}")
# 验证连续性
for _ in range(100):
    eng.环境["光"] = 500; eng.环境["声"] = 50; eng.环境["力"] = 5; eng.环境["温"] = 300
    eng.推进()
    eng2.环境["光"] = 500; eng2.环境["声"] = 50; eng2.环境["力"] = 5; eng2.环境["温"] = 300
    eng2.推进()
match = all(abs(eng.激活[i] - eng2.激活[i]) < 1e-10 for i in range(len(eng.激活)))
print(f"MATCH_AFTER_100={match}")
