#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
候选集"重新生成"的复现性检验：本地开源模型的采样不是逐位可复现的。
同一 prompt、同一 seed、同一模型重新跑一遍，得到的候选集不同。
本脚本把两次独立生成的候选集放在一起比较，给出 rho / top-1 冲突 / 端点位置的
运行间波动 —— 这既是一个诚实的 caveat，也是一条稳健性证据（冲突率稳、rho 不稳）。
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-9


def analyse(df, inst, gap, risk):
    rows = []
    for k, g in df.groupby(inst):
        g = g.dropna(subset=[gap, risk])
        if len(g) < 3 or np.ptp(g[gap]) == 0 or np.ptp(g[risk]) == 0:
            continue
        n = len(g)
        ig, ir = int(np.argmin(g[gap].values)), int(np.argmin(g[risk].values))
        gp = (g[gap].rank(method="average").values - 1) / (n - 1)
        rp = (g[risk].rank(method="average").values - 1) / (n - 1)
        rows.append({"inst": k, "n": n,
                     "rho_ratio": spearmanr(g[gap], g[risk]).statistic,
                     "flip": ig != ir,
                     "gap_pos_min_risk": gp[ir], "risk_pos_min_gap": rp[ig],
                     "gap_min": g[gap].values[ig], "risk_min": g[risk].values[ir],
                     "gap_med": g[gap].median(), "risk_med": g[risk].median()})
    return pd.DataFrame(rows)


runs = {
    ("C slot", "run1"): ("_backup/llm_slot_candidates.csv", "inst", "gap", "overflow"),
    ("C slot", "run2"): ("llm_slot_candidates.csv", "inst", "gap", "overflow"),
    ("D bunker", "run1"): ("_backup/llm_bunker_candidates.csv", "inst", "gap", "overrun"),
    ("D bunker", "run2"): ("llm_bunker_candidates.csv", "inst", "gap", "overrun"),
}

out = []
for (fam, run), (fn, inst, gap, risk) in runs.items():
    if not os.path.exists(os.path.join(OUT, fn)):
        continue
    df = pd.read_csv(os.path.join(OUT, fn))
    df = df[df[gap].notna()]
    A = analyse(df, inst, gap, risk)
    if A.empty:
        continue
    out.append({"family": fam, "run": run, "instances": len(A), "candidates": len(df),
                "rho_ratio": A.rho_ratio.median(), "rho_abs": np.nan,
                "conflict": A.flip.mean(),
                "gap_pos_min_risk": A.gap_pos_min_risk.median(),
                "risk_pos_min_gap": A.risk_pos_min_gap.median(),
                "gap_med": A.gap_med.median(), "risk_med": A.risk_med.median()})
    if len(A) >= 2:
        print(f"{fam} {run}: n_inst={len(A)} conflict={A.flip.mean():.3f} "
              f"rho={A.rho_ratio.median():+.3f} gap_pos(safest)={A.gap_pos_min_risk.median():.2f} "
              f"risk_pos(lowest gap)={A.risk_pos_min_gap.median():.2f}")

R = pd.DataFrame(out)
R.to_csv(os.path.join(OUT, "replication.csv"), index=False)
if not R.empty:
    print("\nrun-to-run spread (max |Δ| across runs, per family):")
    for fam, g in R.groupby("family"):
        d = {c: float(np.nanmax(g[c]) - np.nanmin(g[c])) for c in
             ["conflict", "rho_ratio", "gap_pos_min_risk", "risk_pos_min_gap"]}
        print(f"  {fam}: " + ", ".join(f"{k}={v:.3f}" for k, v in d.items()))
