#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海运语义版第二族：加油港选择（bunker port selection）在承诺制下的超预算风险。

问题：亚欧航线经过 M 个候选加油港。每港有油价 p_j、可加油上限 cap_j、固定港口费 fee_j、
绕航距离 dev_j（绕航油耗 = dev_j * 0.0833 t/nm，按 30t/day、15kn 折算）。本航次需油 B 吨。
    决策 z_j∈{0,1} 是否挂靠加油，y_j∈[0,cap_j] 加油量
    min  Σ p_j y_j + Σ (fee_j + dev_j*0.0833*p_j) z_j
    s.t. Σ y_j >= B ;  y_j <= cap_j z_j
承诺制：对客户的运费/燃油条款已锁定（预算 = 名义最优成本），事后不得追加燃油附加费。
漂移（不利）：油价上浮、实际需油上浮、港口可用上限下降；缺口只能按紧急现货溢价补（1.5x）。
违约度量：超预算率 = (实际执行成本 - 承诺预算)/承诺预算（连续、可比）。
"""
import json, os, zlib
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
B_DEMAND = 1500.0
BUNK_RATE = 30.0 / 24.0 / 15.0        # t/nm
SPOT_PREMIUM = 1.5
# 名称, 油价(USD/t, 2024 量级), 上限(t), 港口费(USD), 绕航(nm)
PORTS = [("Singapore", 600.0, 1200.0, 15000.0, 60.0),
         ("Fujairah", 590.0, 900.0, 18000.0, 180.0),
         ("PortKlang", 615.0, 800.0, 12000.0, 120.0),
         ("Colombo", 640.0, 600.0, 14000.0, 90.0),
         ("Suez", 660.0, 700.0, 20000.0, 40.0),
         ("Rotterdam", 620.0, 1000.0, 16000.0, 20.0)]
M = len(PORTS)


def gen(seed):
    rng = np.random.default_rng(seed)
    p = np.array([t[1] for t in PORTS]) * (1 + 0.03 * rng.standard_normal(M))
    cap = np.array([t[2] for t in PORTS]) * (1 + 0.10 * rng.standard_normal(M))
    fee = np.array([t[3] for t in PORTS]) * (1 + 0.05 * rng.standard_normal(M))
    dev = np.array([t[4] for t in PORTS])
    B = B_DEMAND * (1 + 0.05 * rng.standard_normal())
    return dict(p=p, cap=cap, fee=fee, dev=dev, B=B)


def fixed_cost(inst, j):
    return inst["fee"][j] + inst["dev"][j] * BUNK_RATE * inst["p"][j]


def solve(inst):
    p, cap, B = inst["p"], inst["cap"], inst["B"]
    f = np.array([fixed_cost(inst, j) for j in range(M)])
    c = np.concatenate([p, f])
    integ = np.concatenate([np.zeros(M), np.ones(M)])
    A = np.hstack([np.eye(M), -np.diag(cap)])          # y_j - cap_j z_j <= 0
    A = np.vstack([A, np.concatenate([-np.ones(M), np.zeros(M)])])  # -Σy <= -B
    b = np.concatenate([np.zeros(M), [-B]])
    res = milp(c=c, constraints=LinearConstraint(A, -np.inf, b), integrality=integ,
               bounds=Bounds(np.concatenate([np.zeros(M), np.zeros(M)]),
                             np.concatenate([cap, np.ones(M)])))
    if not res.success:
        return np.nan, None, None
    y, z = res.x[:M], np.round(res.x[M:]).astype(int)
    return res.fun, z, y


def greedy(inst, key, noise=0.0, rng=None):
    p, cap, B = inst["p"], inst["cap"], inst["B"]
    k = key.copy()
    if noise and rng is not None:
        k = k * (1 + noise * rng.standard_normal(M))
    order = np.argsort(k)
    z = np.zeros(M); y = np.zeros(M)
    for j in order:
        if y.sum() >= B - 1e-9:
            break
        take = min(cap[j], B - y.sum())
        if take > 1e-9:
            z[j] = 1; y[j] = take
    return z, y


def ls(inst, z, y, rng, rounds=4):
    best_c = eval_cost(inst, z, y)
    for _ in range(rounds):
        imp = False
        for j in np.flatnonzero(z == 1):
            for i in np.flatnonzero(z == 0):
                z2, y2 = z.copy(), y.copy()
                moved = y2[j]
                z2[j] = 0; y2[j] = 0
                take = min(inst["cap"][i], moved)
                z2[i] = 1; y2[i] = take
                if moved - take > 1e-9:      # 需要第三港补足，放弃该移动
                    continue
                c2 = eval_cost(inst, z2, y2)
                if c2 < best_c - 1e-9 and y2.sum() >= inst["B"] - 1e-9:
                    z, y, best_c, imp = z2, y2, c2, True
                    break
            if imp:
                break
    return z, y


def eval_cost(inst, z, y):
    return float(np.sum(inst["p"] * y) + np.sum([fixed_cost(inst, j) for j in range(M) if z[j] == 1]))


def candidates(inst, n_cand=9):
    p, cap, dev, B = inst["p"], inst["cap"], inst["dev"], inst["B"]
    keys = {"unit": p + np.array([fixed_cost(inst, j) for j in range(M)]) / np.maximum(cap, 1),
            "price": p, "capdeg": p * (1 + 200.0 / np.maximum(cap, 1)),
            "dev": dev}
    out = {}
    for nm, k in keys.items():
        out[nm] = greedy(inst, k)
    rng = np.random.default_rng(zlib.crc32(str(inst["p"].round(3)).encode()))
    for t in range(4):
        out[f"noise{t}"] = greedy(inst, keys["unit"], noise=0.2 + 0.15 * t, rng=rng)
    z0, y0 = greedy(inst, keys["unit"])
    out["ls"] = ls(inst, z0, y0, rng)
    uniq = {}
    for nm, (z, y) in out.items():
        uniq.setdefault((tuple(z), tuple(np.round(y, 3))), (nm, z, y))
    rows = [(eval_cost(inst, z, y), nm, z, y) for nm, z, y in uniq.values()]
    rows.sort(key=lambda t: t[0])
    return rows[:n_cand]


def realized(inst, z, y, deltas=(0.10, 0.20), K=10):
    """承诺制：港口选择固定；缺口按紧急现货溢价补。
    同时返回 比率口径 与 绝对额口径，用于诊断"分母效应"。"""
    out, absv, fail = [], [], []
    base = eval_cost(inst, z, y)
    for di, dl in enumerate(deltas):
        for k in range(K):
            rng = np.random.default_rng(zlib.crc32(f"{inst['p'].round(4)}|{di}|{k}".encode()))
            p2 = inst["p"] * (1 + dl * rng.uniform(0, 1, M))
            cap2 = inst["cap"] * (1 - dl * rng.uniform(0, 1, M))
            B2 = inst["B"] * (1 + dl * rng.uniform(0, 1))
            got = float(np.minimum(y, cap2 * z).sum())
            short = max(0.0, B2 - got)
            cost2 = float((p2 * np.minimum(y, cap2 * z)).sum()
                          + sum(fixed_cost(inst, j) for j in range(M) if z[j] == 1)
                          + short * SPOT_PREMIUM * p2.max())
            out.append((cost2 - base) / base)
            absv.append(cost2 - base)
            fail.append(float(short > 1e-6))
    return float(np.mean(out)), float(np.mean(absv)), float(np.mean(fail))


inst_rows, cand_rows = [], []
for s in range(40):
    inst = gen(2000 + s)
    opt, zo, yo = solve(inst)
    if not np.isfinite(opt):
        continue
    cs = candidates(inst)
    if len(cs) < 2:
        continue
    gaps, ovr, abss = [], [], []
    for cost, nm, z, y in cs:
        g = (cost - opt) / opt
        v, av, f = realized(inst, z, y)
        gaps.append(g); ovr.append(v); abss.append(av)
        cand_rows.append({"inst": 2000 + s, "name": nm, "cost": cost, "gap": g,
                          "overrun": v, "abs_overrun": av, "hard_fail": f,
                          "ports": "+".join(PORTS[j][0] for j in range(M) if z[j] == 1),
                          "fill": float(y.sum() / inst["B"])})
    gaps, ovr, abss = np.array(gaps), np.array(ovr), np.array(abss)
    rho = spearmanr(gaps, ovr).statistic if np.ptp(ovr) > 0 else np.nan
    rho_abs = spearmanr(gaps, abss).statistic if np.ptp(abss) > 0 else np.nan
    ig, iv = int(np.argmin(gaps)), int(np.argmin(ovr))
    inst_rows.append({"inst": 2000 + s, "n_cand": len(cs), "opt": opt, "rho": rho,
                      "rho_abs": rho_abs, "flip": bool(ig != iv), "ovr_min_gap": ovr[ig],
                      "ovr_median": float(np.median(ovr)), "ovr_min_ovr": ovr[iv],
                      "abs_min_gap": abss[ig], "abs_median": float(np.median(abss)),
                      "abs_min_ovr": abss[iv],
                      "gap_min_ovr": gaps[iv], "gap_min_gap": gaps[ig],
                      "range": float(np.ptp(ovr))})

I, C = pd.DataFrame(inst_rows), pd.DataFrame(cand_rows)
I.to_csv(os.path.join(OUT, "sea_bunker_instances.csv"), index=False)
C.to_csv(os.path.join(OUT, "sea_bunker_candidates.csv"), index=False)

J = I[I["range"] > 1e-6]
print(f"实例 {len(I)}（有差异 {len(J)}），候选 {len(C)}")
print(f"top-1 反转率: {I.flip.mean():.3f}")
print(f"rho(gap, 超预算率) 中位: {I.rho.median():.3f}")
print(f"rho(gap, 超预算绝对额) 中位: {I.rho_abs.median():.3f}   <- 若符号翻转/归零，说明比率口径有分母效应")
print(f"min-gap 方案超预算率 中位 = {J.ovr_min_gap.median():.4f}")
print(f"全体候选  超预算率 中位 = {J.ovr_median.median():.4f}")
print(f"最稳方案  超预算率 中位 = {J.ovr_min_ovr.median():.4f}")
print(f"min-gap 方案绝对超支 中位 = {J.abs_min_gap.median():.0f} / 全体 {J.abs_median.median():.0f} / 最稳 {J.abs_min_ovr.median():.0f}")
print(f"最稳方案的 gap 中位 = {J.gap_min_ovr.median():.4f}")
print(f"min-gap 方案的 gap 中位 = {J.gap_min_gap.median():.4f}")
best = C[C.inst.isin(J.inst)]
print("\n最常出现在 min-gap / min-overrun 的港口组合:")
for tag, col in (("min-gap", "gap"), ("min-overrun", "overrun")):
    picks = C.loc[C.groupby("inst")[col].idxmin()]
    print(f"  {tag}: " + ", ".join(picks.ports.value_counts().head(3).index.tolist()))
print(f"\n最小 gap 方案的加油完成度 中位 = {C.loc[C.groupby('inst').gap.idxmin()].fill.median():.3f}")
print(f"最小超预算方案的加油完成度 中位 = {C.loc[C.groupby('inst').overrun.idxmin()].fill.median():.3f}")
