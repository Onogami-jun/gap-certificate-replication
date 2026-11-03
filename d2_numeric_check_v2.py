#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D2 数值验证 v2 —— 改用"支付意愿"而不是猜成本参数。
第一版的问题：kc/ks 区间没标定，认证在任何点都不值得（全 0），看不到相变。

这里直接算两个 WTP（都以"各自最优影子配置"为基准，认证与影子互为替代）：
  WTP_cert(q)   = [c*(ncert=0) - c*(ncert=2)] / 2    每家的可支付认证成本
  WTP_shadow(q) = c*(ncert=0, K=0) - c*(ncert=0, K*)  影子产能的净价值
若 regime B（认证管不了共同灾害）下 WTP_cert 随 q 严格降到 0，而 regime A 不降，
则"相变由相关性驱动"成立。再看零点 q_hat 是否随边际中断概率 p 移动。
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
CAP, D, ALPHA = 0.5, 1.0, 0.4
KS, PIS = 0.10, [2.0, 5.0, 10.0]   # 影子单位预付 / 缺货惩罚
P_GRID = [0.05, 0.15, 0.30]


def joint(p1, p2, q):
    q = min(q, p1, p2)
    r1 = (p1 - q) / (1 - q) if q < 1 else 0.0
    r2 = (p2 - q) / (1 - q) if q < 1 else 0.0
    return {2: (1 - q) * (1 - r1) * (1 - r2),
            1: (1 - q) * (r1 * (1 - r2) + (1 - r1) * r2),
            0: q + (1 - q) * r1 * r2}


def e_short(P, K):
    return sum(pr * max(0.0, D - CAP * u - K) for u, pr in P.items())


def cert_par(regime, p, q):
    if regime == "A":
        pc = ALPHA * p
        qq = min(ALPHA * q, pc)
    else:
        qq = min(q, p)
        r = (p - qq) / (1 - qq)
        pc = qq + ALPHA * r * (1 - qq)
    return pc, qq


def opt_K(P, ks, pi, kc=0.0):
    best = (1e18, None)
    for K in (0.0, CAP, 1.0):
        c = kc + ks * K + pi * e_short(P, K)
        if c < best[0]:
            best = (c, K)
    return best


rows = []
for regime in ("A", "B"):
    for p in P_GRID:
        for pi in PIS:
            for q in np.linspace(0, p, 41):
                pc, qq = cert_par(regime, p, q)
                P0 = joint(p, p, q)
                P2 = joint(pc, pc, qq)
                c0, K0 = opt_K(P0, KS, pi)
                c0_nosh, _ = opt_K(P0, KS, pi)          # 同上，K 的自由选择
                c0_fixed0 = pi * e_short(P0, 0.0)       # 不用影子
                c2 = 2 * 0.0
                # 全认证：认证成本待求 -> 用差分
                _, K2 = opt_K(P2, KS, pi)
                c2_ops = KS * K2 + pi * e_short(P2, K2)
                rows.append({"regime": regime, "p": p, "pi": pi, "q": round(float(q), 4),
                             "wtp_cert": (c0 - c2_ops) / 2,
                             "wtp_shadow": c0_fixed0 - c0,
                             "K0": K0, "K2": K2})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "d2_wtp.csv"), index=False)

print("== WTP_cert(q)（每家的可支付认证成本），ks=0.10 ==")
for pi in PIS:
    print(f"-- pi={pi}")
    for regime in ("A", "B"):
        for p in P_GRID:
            s = df[(df.regime == regime) & (df.pi == pi) & (df.p == p)].sort_values("q")
            v = s.wtp_cert.values
            qq = s.q.values
            zero = np.nan
            for i in range(len(v) - 1):
                if v[i] > 0 >= v[i + 1]:
                    zero = qq[i] + (qq[i + 1] - qq[i]) * v[i] / (v[i] - v[i + 1])
                    break
            if v[-1] < 0 and np.isnan(zero):
                zero = 0.0
            print(f"   [{regime}] p={p:.2f}: WTP(0)={v[0]:+.3f}  WTP(qmax)={v[-1]:+.3f}  q_hat={zero:.3f}")

print("\n== WTP_shadow(q)：影子价值是否随 q 上升 ==")
for pi in PIS:
    print(f"-- pi={pi}")
    for p in P_GRID:
        s = df[(df.regime == "B") & (df.pi == pi) & (df.p == p)].sort_values("q")
        print(f"   p={p:.2f}: shadowWTP(0)={s.wtp_shadow.values[0]:+.3f} -> (qmax)={s.wtp_shadow.values[-1]:+.3f}")

print("\n== 认证价值对 q 的单调性（regime 对比，pi=5）==")
for regime in ("A", "B"):
    for p in P_GRID:
        s = df[(df.regime == regime) & (df.pi == 5.0) & (df.p == p)].sort_values("q")
        d = np.diff(s.wtp_cert.values)
        print(f"   [{regime}] p={p:.2f}: WTP 从 {s.wtp_cert.values[0]:+.3f} 到 {s.wtp_cert.values[-1]:+.3f}，"
              f"递减步数 {int((d < -1e-9).sum())}/{len(d)}")
