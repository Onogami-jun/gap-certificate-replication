#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
第二轮审查：上一轮把"实现不可行"用单边修复吸收了，这可能人为压平了排序差异。
这里改用最自然的量——**实现违约率**（realized infeasibility rate）——重做排序对比，
不引入任何修复规则、不编造罚值。

对每个 nominal-feasible 候选 x 与每个情景 q：
    viol(x) = P_q[ x 在 q 下不可行 ]
然后比较 gap 排序 vs viol 排序。
"""
import json, os, sys, zlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXP = r"C:\Users\D0gSXG\Desktop\project\论文\paper_dual_gap\experiments"
sys.path.insert(0, EXP)
import run_real as R  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(EXP, "results_v2", "candidates_labeled.csv")
BASE_SEED = {"Knapsack": 1000, "Portfolio": 2000, "Diet": 3000, "SetCover": 4000,
             "BinPacking": 5000, "TSP": 6000, "Scheduling": 7000, "Facility": 8000}
TOL = 1e-6


def get_problem(ptype, inst):
    seed = int(str(inst).split("_s")[-1])
    return R.GENERATORS[ptype](seed - BASE_SEED[ptype])


def perturb(p, ptype, delta, rng):
    """不利漂移；返回值是"实现后"的判定参数（不构造新 Problem）。"""
    if ptype == "Knapsack":
        return ("A_ub", p.A_ub * (1 + delta * rng.uniform(0, 1, p.A_ub.shape)))
    if ptype == "Portfolio":
        return ("A_ub", p.A_ub[0:1, :] * (1 + delta * rng.uniform(0, 1, (1, p.n))))
    if ptype == "Diet":
        return ("A_ub", p.A_ub * (1 - delta * rng.uniform(0, 1, p.A_ub.shape)))
    if ptype == "SetCover":
        A = -p.A_ub.copy()
        drop = (rng.random(A.shape) < delta) & (A > 0)
        A2 = A * (~drop)
        for i in range(A2.shape[0]):
            if A2[i].sum() == 0:
                A2[i, int(np.flatnonzero(A[i])[rng.integers(int(A[i].sum()))])] = 1.0
        return ("A_ub", -A2)
    return ("none", None)


def violated(p, ptype, x, kind, A):
    if kind == "none":
        return False
    lhs = A @ x
    return bool(np.any(lhs > p.b_ub + TOL))


rows, inst_rows = [], []
df = pd.read_csv(CSV)
feas = df[df["feasible"] == True]  # noqa: E712
n_infeasible_raw = int((df["feasible"] != True).sum())  # noqa: E712

for (ptype, inst), g in feas.groupby(["problem_type", "instance"], sort=False):
    p = get_problem(ptype, inst)
    xs, seen = [], set()
    for _, r in g.iterrows():
        x = np.array(json.loads(r["x_json"]), float)
        if len(x) != p.n:
            continue
        key = tuple(np.round(x, 6))
        if key in seen:
            continue
        seen.add(key)
        xs.append((x, float(r["gap"])))
    if len(xs) < 2:
        continue

    deltas, K = [0.10, 0.20], 10
    v = np.zeros((len(xs), len(deltas), K))
    for di, d in enumerate(deltas):
        for k in range(K):
            rng = np.random.default_rng(zlib.crc32(f"{inst}|{di}|{k}".encode()))
            kind, A = perturb(p, ptype, d, rng)
            for ci, (x, _) in enumerate(xs):
                v[ci, di, k] = violated(p, ptype, x, kind, A)
    viol = v.mean(axis=(1, 2))
    gaps = np.array([t[1] for t in xs])
    rho = spearmanr(gaps, viol).statistic if np.ptp(viol) > 0 else np.nan
    i_g, i_v = int(np.argmin(gaps)), int(np.argmin(viol))
    inst_rows.append({
        "problem_type": ptype, "instance": inst, "n_cand": len(xs),
        "rho_gap_viol": rho,
        "flip": bool(i_g != i_v),
        "viol_min_gap": float(viol[i_g]), "viol_max_gap": float(viol[int(np.argmax(gaps))]),
        "viol_min_viol": float(viol[i_v]), "gap_min_viol": float(gaps[i_v]),
        "range_viol": float(np.ptp(viol)),
    })
    for ci, (x, gp) in enumerate(xs):
        rows.append({"problem_type": ptype, "instance": inst, "cand": ci,
                     "gap": gp, "viol_rate": float(viol[ci])})

I = pd.DataFrame(inst_rows)
C = pd.DataFrame(rows)
I.to_csv(os.path.join(OUT, "audit2_instances.csv"), index=False)
C.to_csv(os.path.join(OUT, "audit2_candidates.csv"), index=False)

print(f"nominal 不可行候选占比: {n_infeasible_raw}/{len(df)} = {n_infeasible_raw/len(df):.1%}")
print(f"可排序实例: {len(I)}\n")
summ = I.groupby("problem_type").apply(lambda g: pd.Series({
    "n": len(g),
    "flip_rate": g["flip"].mean(),
    "median_rho": g["rho_gap_viol"].median(),
    "viol_at_min_gap": g["viol_min_gap"].median(),
    "viol_at_max_gap": g["viol_max_gap"].median(),
    "share_viol_range>0": (g["range_viol"] > 0).mean(),
}), include_groups=False).round(3)
print(summ.to_string())
print("\n总体: flip=%.3f  median_rho=%.3f" % (I["flip"].mean(), I["rho_gap_viol"].median()))
json.dump({k: v for k, v in summ.T.to_dict().items()}, open(os.path.join(OUT, "audit2_summary.json"), "w"), ensure_ascii=False, indent=2)
