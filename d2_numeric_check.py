#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2 数值验证：认证 vs 影子产能，相变是否由"中断相关性"驱动（而非边际中断概率）。

模型
----
n=2 供应商，各产能 cap=0.5，需求 d=1（单源不足，双源刚好满足）。
中断结构 = 共同冲击 + 独立冲击：
    P(共同冲击)=q（发生则全部中断）；否则供应商 i 独立中断概率 r_i，满足 p_i = q + r_i(1-q)。
认证成本 k_c / 家；认证把中断概率降为 p_c。
  设定 A（认证降全部风险，q、r 等比缩）：q->αq, r->αr
  设定 B（认证只管"自身管理"，不防共同灾害）：r->αr, q 不变
影子产能：预付 k_s·K（K∈[0,1]），中断时可用；缺货惩罚 π=1 / 单位。
决策：认证几家 S∈{0,1,2} × 影子容量 K。
关键：缺货是关于 K 的分段线性凸函数，最优 K 必在断点 {0, 0.5, 1}。
"""
import json, os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
CAP, D, PI, ALPHA = 0.5, 1.0, 1.0, 0.4
KS_GRID = np.round(np.arange(0.0, 0.5001, 0.01), 3)
KC_GRID = [0.05, 0.10, 0.20]
P_GRID = [0.05, 0.15, 0.30]


def joint(p1, p2, q):
    """返回 {(u1,u2): prob}，u=1 表示未中断。"""
    q = min(q, p1, p2)
    r1 = (p1 - q) / (1 - q) if q < 1 else 0.0
    r2 = (p2 - q) / (1 - q) if q < 1 else 0.0
    return {2: (1 - q) * (1 - r1) * (1 - r2),
            1: (1 - q) * (r1 * (1 - r2) + (1 - r1) * r2),
            0: q + (1 - q) * r1 * r2}


def shortage(P, K):
    return sum(pr * max(0.0, D - CAP * u - K) for u, pr in P.items())


def best(regime, p, q, kc, ks, ncert):
    if regime == "A":
        pc = ALPHA * p
        qq = min(ALPHA * q, pc)
    else:  # B：只降独立部分
        qq = min(q, p)
        r = (p - qq) / (1 - qq)
        pc = qq + ALPHA * r * (1 - qq)
    if ncert == 0:
        P = joint(p, p, q)
    elif ncert == 1:
        P = joint(pc, p, qq)
    else:
        P = joint(pc, pc, qq)
    vals = [(kk, ncert * kc + ks * K + PI * shortage(P, K)) for K in (0.0, CAP, 1.0) for kk in (K,)]
    K, c = min(vals, key=lambda t: t[1])
    return c, K


rows = []
for regime in ("A", "B"):
    for p in P_GRID:
        for kc in KC_GRID:
            for ks in (0.05, 0.10, 0.20):
                qs = np.linspace(0, p, 41)
                for q in qs:
                    c0, K0 = best(regime, p, q, kc, ks, 0)
                    c1, K1 = best(regime, p, q, kc, ks, 1)
                    c2, K2 = best(regime, p, q, kc, ks, 2)
                    n = min({0: c0, 1: c1, 2: c2}, key=lambda i: [c0, c1, c2][i])
                    rows.append({"regime": regime, "p": p, "q": round(float(q), 4), "kc": kc, "ks": ks,
                                 "nstar": n, "Kstar": [K0, K1, K2][n],
                                 "c0": c0, "c1": c1, "c2": c2})

df = pd.DataFrame(rows).drop_duplicates(subset=["regime", "p", "q", "kc", "ks"])
df.to_csv(os.path.join(OUT, "d2_grid.csv"), index=False)

# ---- 1. 三区是否存在 ----
print("== 最优认证家数 nstar 的取值分布 ==")
print(df.groupby(["regime", "nstar"]).size().unstack(fill_value=0).to_string())

# ---- 2. 认证价值 V(q)=c0-c2 的零点，是否随 p 移动 ----
print("\n== 全认证的价值 V(q)=c0-c2，零点 q_hat（ks=0.10）==")
for regime in ("A", "B"):
    for kc in KC_GRID:
        line = []
        for p in P_GRID:
            sub = df[(df.regime == regime) & (df.p == p) & (df.kc == kc) & (df.ks == 0.10)].sort_values("q")
            V = (sub.c0 - sub.c2).values
            qq = sub.q.values
            zero = np.nan
            for i in range(len(V) - 1):
                if V[i] > 0 >= V[i + 1]:
                    zero = qq[i] + (qq[i + 1] - qq[i]) * V[i] / (V[i] - V[i + 1])
                    break
            line.append(f"p={p:.2f}: V(0)={V[0]:.4f} V(max)={V[-1]:.4f} q_hat={zero:.3f}" if not np.isnan(V[0]) else "")
        print(f"  [{regime}] kc={kc}")
        for s in line:
            print("     ", s)

# ---- 3. 影子价值 W(q)=c0(min K) - c0(K=0) 随 q 的斜率 ----
print("\n== 影子产能相对价值随 q 的变化（regime B, kc=0.10, ks=0.10）==")
for p in P_GRID:
    sub = df[(df.regime == "B") & (df.p == p) & (df.kc == 0.10) & (df.ks == 0.10)].sort_values("q")
    if sub.empty: continue
    # 用 c0 情景对比：K=0 与最优 K
    print(f"  p={p:.2f}: nstar(q) = " + "".join(str(int(v)) for v in sub.nstar.values))

# ---- 4. 相变边界 (q, ks) 平面上的 nstar ----
print("\n== (q 行, ks 列) 分区表：regime B, p=0.15, kc=0.10 ==")
sub = df[(df.regime == "B") & (df.p == 0.15) & (df.kc == 0.10)]
tab = sub.pivot_table(index="q", columns="ks", values="nstar", aggfunc="first")
ks_show = [k for k in tab.columns if abs((k * 100) % 5) < 1e-9][:12]
print(tab[ks_show].to_string())

json.dump({"rows": len(df)}, open(os.path.join(OUT, "d2_meta.json"), "w"))
print("\n(saved d2_grid.csv)")
