#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""等值池实验（scale-up）：目标值几乎相同、承诺量不同的方案池里，gap 能否分辨暴露。

构造：在最优解上做等收益替换（两笔订舱的 r·q 差 < 0.1% 最优毛利），
使池内 gap 跨度 ≤ 0.1 pp，而承诺箱量不同；再对每个方案按同一燃油方案评估暴露。
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import real_case as rc

BOOKS = list(range(201, 221))
N_SCEN = 60
TIE = 0.001          # 0.1 pp of optimal contribution


def feasible(book, x, penalty):
    caps = rc.CAP_TEU.copy()
    caps[3] = max(1000.0, caps[3] - penalty)
    load = (book["w"] * (book["q"] * x)[:, None]).sum(0)
    return bool(np.all(load <= caps + 1e-6))


rows = []
for seed in BOOKS:
    book = rc.gen_book(seed)
    grid_sg = np.linspace(rc.B_SG_LEG, rc.TANK["Singapore"], 5)
    grid_rt = np.linspace(0, rc.TANK["Rotterdam"], 4)
    obj0, x0, ysg, yrt = rc.joint_opt(book, grid_sg, grid_rt)
    if x0 is None:
        continue
    pen = rc.fuel_draft_penalty(ysg, yrt)
    rev0 = float(book["r"] * book["q"] @ x0)

    S = set(np.flatnonzero(x0 == 1).tolist())
    cand = []
    for i in sorted(S):
        for j in np.flatnonzero(x0 == 0):
            y = x0.copy()
            y[i], y[j] = 0, 1
            if not feasible(book, y, pen):
                continue
            drew = float(book["r"][j] * book["q"][j] - book["r"][i] * book["q"][i])
            if abs(drew) > TIE * abs(rev0):
                continue
            cand.append((float(book["q"] @ y), y, drew))
    if len(cand) < 3:
        rows.append(dict(seed=seed, n_pool=0))
        continue

    cand.sort(key=lambda t: t[0])
    k = min(5, len(cand))
    pick = [cand[int(round(m * (len(cand) - 1) / max(1, k - 1)))] for m in range(k)]
    plans = [(rev0, x0)] + [(rev0 + d, y) for _, y, d in pick]
    objv = [r - (rc.VLSFO["Singapore"] * ysg + rc.VLSFO["Rotterdam"] * yrt) for r, _ in plans]
    base = max(objv)
    gap = np.array([100 * (base - o) / abs(base) for o in objv])
    teu = np.array([float(book["q"] @ y) for _, y in plans])
    expo = np.array([100 * rc.risk_of(book, y, ysg, yrt, N_SCEN)[0] for _, y in plans])
    rho = spearmanr(gap, expo).statistic if np.std(gap) > 1e-9 else np.nan
    rows.append(dict(seed=seed, n_pool=len(plans),
                     gap_spread=round(float(gap.max() - gap.min()), 4),
                     teu_spread=round(float(teu.max() - teu.min()), 0),
                     expo_spread=round(float(expo.max() - expo.min()), 3),
                     expo_min=round(float(expo.min()), 2), expo_max=round(float(expo.max()), 2),
                     rho=round(float(rho), 3) if rho == rho else np.nan))

d = pd.DataFrame(rows)
d.to_csv("tie_pool.csv", index=False)
ok = d[d.n_pool > 0]
print(d.to_string(index=False))
print("\npools built: %d / %d" % (len(ok), len(d)))
print("median gap spread %.4f pp | median exposure spread %.2f pp"
      % (ok.gap_spread.median(), ok.expo_spread.median()))
print("pools with gap spread < 0.1 pp and exposure spread > 1 pp: %d"
      % int(((ok.gap_spread < 0.1) & (ok.expo_spread > 1.0)).sum()))
print("median TEU spread %.0f" % ok.teu_spread.median())
print("share of pools with rho < 0: %.2f" % float((ok.rho < 0).mean()))
