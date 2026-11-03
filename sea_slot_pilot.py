#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海运语义版：多航段舱位分配（slot allocation / overbooking）在承诺制下的脆弱性。

问题（真实结构）：一条航线 K 个航段，每段舱位 cap_k TEU。有 N 个订舱请求，
每个请求占 [o_i, d_i) 的航段、箱量 q_i、运价 r_i。决策 = 接受哪些请求（0/1）。
    目标 max Σ r_i x_i
    约束 Σ_{i 覆盖 k} q_i x_i <= cap_k   （对每个航段）
这是多维背包（MCKP）的真实海运形态：舱位紧张时最优解必然贴满每个航段。

承诺制：接受订舱 = 对客户承诺舱位，发出后不可撤回（甩柜有罚金/信誉损失）。
漂移（不利）：实际箱量上浮（超订、箱型不符）、航段有效容量下降（拥堵、吃水、空箱占位）。
违约：存在某航段 Σ q_i' > cap_k'  → 承诺无法全部兑现。

检验：gap 排序 vs 违约率排序是否反转；最小 gap 的方案是不是最脆的那个。
"""
import json, os, zlib
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
N_LEGS = 4


def gen_slot(seed, n_book=22, cap=8000, tight=1.18):
    """生成一个舱位分配实例；tight 控制总需求/容量的比（>1 表示舱位紧张）。"""
    rng = np.random.default_rng(seed)
    o = rng.integers(0, N_LEGS, n_book)
    d = np.array([rng.integers(oo + 1, N_LEGS + 1) for oo in o])
    q = rng.uniform(80, 600, n_book)
    # 归一化使加权需求约为 cap*tight
    w = np.zeros((n_book, N_LEGS))
    for i in range(n_book):
        w[i, o[i]:d[i]] = 1.0
    load = (w * q[:, None]).sum(axis=0)
    q = q * (cap * tight / load.max())
    r = rng.uniform(900, 2600, n_book)
    caps = np.full(N_LEGS, float(cap))
    c = -r                                   # 最大化 -> 最小化负收益
    A = (w * q[:, None] / caps[None, :]).T   # (航段, 订舱)，A@x <= 1 即容量约束
    return dict(c=c, A=A, b=np.ones(N_LEGS), q=q, r=r, w=w, caps=caps,
                o=o, d=d, n=n_book, opt=np.nan)


def solve(inst):
    res = milp(c=inst["c"], constraints=LinearConstraint(inst["A"], -np.inf, inst["b"]),
               integrality=np.ones(inst["n"]), bounds=Bounds(0, 1))
    if not res.success:
        return np.nan, None
    return -res.fun, np.round(res.x).astype(int)


def greedy(inst, key, rng=None, noise=0.0):
    n = inst["n"]
    k = key.copy()
    if noise and rng is not None:
        k = k * (1 + noise * rng.standard_normal(n))
    order = np.argsort(-k)
    x = np.zeros(n)
    load = np.zeros(N_LEGS)
    for i in order:
        if np.all(load + inst["A"][:, i] <= 1 + 1e-12):
            x[i] = 1
            load += inst["A"][:, i]
    return x


def local_search(inst, x, rng, rounds=3):
    x = x.copy()
    for _ in range(rounds):
        improved = False
        for i in np.flatnonzero(x == 1):
            for j in np.flatnonzero(x == 0):
                y = x.copy(); y[i], y[j] = 0, 1
                if np.all(inst["A"] @ y <= 1 + 1e-12) and inst["r"] @ y > inst["r"] @ x + 1e-9:
                    x = y; improved = True; break
            if improved:
                break
    return x


def candidates(inst, n_cand=10):
    dens = inst["r"] / inst["q"]
    cands = {}
    for name, key in (("dens", dens), ("rev", inst["r"]), ("small", -inst["q"])):
        cands[name] = greedy(inst, key)
    rng = np.random.default_rng(zlib.crc32(str(inst["o"].tolist()).encode()))
    for t in range(6):
        cands[f"noise{t}"] = greedy(inst, dens, rng, noise=0.35 + 0.15 * t)
    cands["ls"] = local_search(inst, cands["dens"], rng)
    # 去重，按目标值排序后保留 10 个（覆盖好/中/差）
    uniq = {}
    for k, v in cands.items():
        uniq.setdefault(tuple(np.asarray(v, float)), k)
    items = [(float(inst["r"] @ np.array(t, float)), k, np.array(t, float))
             for t, k in uniq.items()]
    items.sort(key=lambda t: -t[0])
    return items[:n_cand]


def viol_rate(inst, x, deltas=(0.10, 0.20), K=10):
    """返回 (平均甩柜率, 硬违约率)。甩柜率 = 超出舱位量 / 已承诺总量，连续可比。"""
    ov, ab, bad = [], [], []
    for di, dl in enumerate(deltas):
        for k in range(K):
            rng = np.random.default_rng(zlib.crc32(f"{inst['o'].tolist()}|{di}|{k}".encode()))
            q2 = inst["q"] * (1 + dl * rng.uniform(0, 1, inst["n"]))
            cap2 = inst["caps"] * (1 - dl * rng.uniform(0, 1, N_LEGS))
            load = (inst["w"] * (q2 * x)[:, None]).sum(axis=0)
            over = np.maximum(0.0, load - cap2)
            ov.append(float(over.max() / max(1e-9, (inst["q"] * x).sum())))
            ab.append(float(over.max()))
            bad.append(float(np.any(over > 1e-9)))
    return float(np.mean(ov)), float(np.mean(ab)), float(np.mean(bad))


inst_rows, cand_rows = [], []
for s in range(40):
    inst = gen_slot(1000 + s, n_book=int(18 + 6 * (s % 3)), cap=8000,
                    tight=[1.05, 1.15, 1.30][s % 3])
    opt, xopt = solve(inst)
    if not np.isfinite(opt):
        continue
    inst["opt"] = opt
    cs = candidates(inst)
    if len(cs) < 2:
        continue
    gaps, viols, abss, bads, vals = [], [], [], [], []
    for val, name, x in cs:
        g = (opt - val) / abs(opt)
        v, av, b = viol_rate(inst, x)
        gaps.append(g); viols.append(v); abss.append(av); bads.append(b); vals.append(val)
        cand_rows.append({"inst": 1000 + s, "name": name, "obj": val, "gap": g,
                          "overflow": v, "abs_overflow": av, "hard_viol": b,
                          "util": float(np.max(inst["A"] @ x))})
    gaps, viols, abss = np.array(gaps), np.array(viols), np.array(abss)
    rho = spearmanr(gaps, viols).statistic if np.ptp(viols) > 0 else np.nan
    rho_abs = spearmanr(gaps, abss).statistic if np.ptp(abss) > 0 else np.nan
    ig, iv = int(np.argmin(gaps)), int(np.argmin(viols))
    inst_rows.append({"inst": 1000 + s, "n_cand": len(cs), "opt": opt,
                      "rho_gap_viol": rho, "rho_abs": rho_abs, "flip": bool(ig != iv),
                      "viol_min_gap": viols[ig], "viol_median": float(np.median(viols)),
                      "viol_min_viol": viols[iv], "gap_min_viol": gaps[iv],
                      "gap_min_gap": gaps[ig], "util_min_gap": float(np.max(inst["A"] @ cs[ig][2])),
                      "abs_min_gap": abss[ig], "abs_median": float(np.median(abss)),
                      "abs_min_viol": abss[iv],
                      "hard_min_gap": bads[ig], "hard_median": float(np.median(bads)),
                      "range_viol": float(np.ptp(viols))})

I, C = pd.DataFrame(inst_rows), pd.DataFrame(cand_rows)
I.to_csv(os.path.join(OUT, "sea_slot_instances.csv"), index=False)
C.to_csv(os.path.join(OUT, "sea_slot_candidates.csv"), index=False)

J = I[I.range_viol > 1e-6]
print(f"实例 {len(I)}，甩柜率有差异 {len(J)}；候选 {len(C)}")
print(f"top-1 反转率: {I.flip.mean():.3f}（全部） / {J.flip.mean():.3f}（有差异）")
print(f"rho(gap, 甩柜率) 中位: {I.rho_gap_viol.median():.3f}  /  rho(gap, 绝对甩柜量): {I.rho_abs.median():.3f}")
print("\n最小 gap 的方案 vs 其他:")
print(f"  min-gap 甩柜率 中位 = {J.viol_min_gap.median():.4f}")
print(f"  全体候选 甩柜率 中位 = {J.viol_median.median():.4f}")
print(f"  最低甩柜率方案 甩柜率 中位 = {J.viol_min_viol.median():.4f}")
print(f"  min-gap 方案的舱位利用率 中位 = {J.util_min_gap.median():.3f}")
print(f"  min-viol 方案的 gap 中位 = {J.gap_min_viol.median():.3f}")
print(f"  min-gap 方案硬违约率 中位 = {J.hard_min_gap.median():.3f} / 全体 = {J.hard_median.median():.3f}")
print("\n按需求/容量比分组:")
I["tight"] = I.inst.map(lambda v: [1.00, 1.10, 1.25][(int(v) - 1000) % 3])
print(I.groupby("tight").agg(n=("inst", "size"), flip=("flip", "mean"),
                             rho=("rho_gap_viol", "median"),
                             viol_min_gap=("viol_min_gap", "median"),
                             viol_med=("viol_median", "median")).round(3).to_string())
