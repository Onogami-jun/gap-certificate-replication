#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2 数值验证 v3 —— v2 暴露了一个结构性问题：
两供应商 + 线性缺货惩罚时 E[缺货] = p（与相关性 q 无关），
所以"相关性"根本不进目标函数，相变不可能由它驱动。

本版加凸缺货成本（部分缺货的边际损失递增，现实中成立），
使 #up 的分布（即相关性）真正进入目标，再重测：
  WTP_cert(q) 是否在 regime B（认证管不了共同灾害）下严格降到 0，
  而 regime A（认证能降共同冲击）下基本不降。
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
CAP, D, ALPHA, KS = 0.5, 1.0, 0.4, 0.10
GAMMA = 2.0                     # 缺货成本 pi * s^GAMMA（凸）
P_GRID = [0.05, 0.15, 0.30]
PIS = [1.0, 2.0, 4.0]
KGRID = np.round(np.arange(0, 1.0001, 0.02), 3)


def joint(p1, p2, q):
    q = min(q, p1, p2)
    r1 = (p1 - q) / (1 - q) if q < 1 else 0.0
    r2 = (p2 - q) / (1 - q) if q < 1 else 0.0
    return {2: (1 - q) * (1 - r1) * (1 - r2),
            1: (1 - q) * (r1 * (1 - r2) + (1 - r1) * r2),
            0: q + (1 - q) * r1 * r2}


def e_cost(P, K, pi):
    return sum(pr * pi * max(0.0, D - CAP * u - K) ** GAMMA for u, pr in P.items())


def opt_K(P, pi):
    vals = [(KS * K + e_cost(P, K, pi), K) for K in KGRID]
    return min(vals, key=lambda t: t[0])


def cert_par(regime, p, q):
    if regime == "A":
        pc = ALPHA * p
        qq = min(ALPHA * q, pc)
    else:
        qq = min(q, p)
        r = (p - qq) / (1 - qq)
        pc = qq + ALPHA * r * (1 - qq)
    return pc, qq


rows = []
for regime in ("A", "B"):
    for p in P_GRID:
        for pi in PIS:
            for q in np.linspace(0, p, 41):
                pc, qq = cert_par(regime, p, q)
                P0, P2 = joint(p, p, q), joint(pc, pc, qq)
                c0, K0 = opt_K(P0, pi)
                c2, K2 = opt_K(P2, pi)
                rows.append({"regime": regime, "p": p, "pi": pi, "q": round(float(q), 4),
                             "wtp_cert": (c0 - c2) / 2, "K0": K0, "K2": K2,
                             "e0": e_cost(P0, K0, pi), "e2": e_cost(P2, K2, pi)})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "d2_wtp_v3.csv"), index=False)

print(f"凸缺货成本 gamma={GAMMA}, ks={KS}\n== WTP_cert(q) ==")
for pi in PIS:
    print(f"-- pi={pi}")
    for regime in ("A", "B"):
        for p in P_GRID:
            s = df[(df.regime == regime) & (df.pi == pi) & (df.p == p)].sort_values("q")
            v, qq = s.wtp_cert.values, s.q.values
            zero = np.nan
            for i in range(len(v) - 1):
                if v[i] > 0 >= v[i + 1]:
                    zero = qq[i] + (qq[i + 1] - qq[i]) * v[i] / (v[i] - v[i + 1])
                    break
            print(f"   [{regime}] p={p:.2f}: WTP(0)={v[0]:+.4f} WTP(qmax)={v[-1]:+.4f} q_hat={zero:.3f}")

print("\n== A 与 B 的 WTP 差（应为正且随 q 增大）: WTP_A - WTP_B ==")
for pi in PIS:
    for p in P_GRID:
        a = df[(df.regime == "A") & (df.pi == pi) & (df.p == p)].sort_values("q").wtp_cert.values
        b = df[(df.regime == "B") & (df.pi == pi) & (df.p == p)].sort_values("q").wtp_cert.values
        print(f"   pi={pi} p={p:.2f}: q=0 -> {a[0]-b[0]:+.4f} ; q=qmax -> {a[-1]-b[-1]:+.4f}")

print("\n== 影子配置是否随 q 变（regime B, pi=2）：K* ==")
for p in P_GRID:
    s = df[(df.regime == "B") & (df.pi == 2.0) & (df.p == p)].sort_values("q")
    print(f"   p={p:.2f}: " + " ".join(f"{k:.2f}" for k in s.K0.values[::8]))
