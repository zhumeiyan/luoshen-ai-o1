import sys
sys.path.insert(0, "/mnt/data/catpaw/home/workspace/洛神赋1.1")
from 活诊断 import 提取锚点, 归一化
mouth_vals, aud_vals, n = 提取锚点()
mouth_norm, m_lo, m_hi = 归一化(mouth_vals)
aud_norm, a_lo, a_hi = 归一化(aud_vals)

m = 0.3
a = 0.3
N_M = len(mouth_norm)
N_A = len(aud_norm)

with open("cycle_dump_1000.txt", "w") as f:
    for cycle in range(1000):
        for i in range(56):
            ext_m = mouth_norm[i % N_M]
            ext_a = aud_norm[i % N_A]
            m_new = 0.46 * m + 0.22 * a + 0.32 * ext_m
            a_new = 0.46 * a + 0.22 * m + 0.32 * ext_a
            m = m_new
            a = a_new
        f.write(f"cycle {cycle:4d}  m={m:.16f}  a={a:.16f}\n")

print("done — cycle_dump_1000.txt written")
