#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4.2 节的形式化：承诺量杠杆 (commitment lever) 下的"端点锁定"命题与冲突率下界。

命题 1（Lever opposition ⇒ forced conflict）
  设候选集 X，标量杠杆 v: X -> R。若
    (A1) gap 在 v 上严格单调；
    (A2) 风险 R 在 v 上严格单调；
    (A3) 两者的单调方向相反（dgap/dv * dR/dv < 0），
  则 argmin gap 与 argmin R 必落在 v 的两个不同端点；只要候选集在 v 上有 ≥2 个取值，
  两个 argmin 就是不同的方案，top-1 冲突恒为 1。
  证明：单调性把 argmin gap 钉在 v 的一个端点，把 argmin R 钉在另一端；端点不同 ⇒ 方案不同。∎

推论 1（单调重标定不能救）
  若存在 x,y 使 v(x)>v(y)、gap(x)≤gap(y)、R(x)≥R(y)（至少一个严格），
  则不存在非降函数 φ 使 φ(gap(x))=R(x) 对所有 x 成立。
  故"换归一化/换标尺"不可能修复 gap 作为风险代理；比率口径与绝对口径的差别只是二阶。
  证明：单调 φ 保序，与 R 的反序矛盾。∎

推论 2（冲突率下界）
  记 Λ(ω)=1 当且仅当实例内 v 的排序与 gap 排序、风险排序同时严格反向；
  记 EOR(ω)=1 当且仅当 v 最大者取到最小 gap 且 v 最小者取到最小风险。
  则 ConflictRate(Ω) ≥ E[Λ] ≥ E[EOR]（两者均由命题 1 直接推得），
  残余冲突来自杠杆之外的异质性（价格组合、港口组合等）。

本脚本用已有 LLM 候选数据做数值验证：
  C 族杠杆 v = 已承诺箱量 committed_teu（越大 → gap 越小、甩柜风险越大）
  D 族杠杆 v = 加油覆盖率 fill = lift/B（越大 → gap 越大、超支风险越小）
并跑一个受控合成实验，量化"残余异质性 → 冲突率下降"的斜率。
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-9


def per_instance(df, inst, gap, risk, lever, min_n=3):
    rows = []
    for k, g in df.groupby(inst):
        g = g.dropna(subset=[gap, risk, lever])
        if len(g) < min_n:
            continue
        if np.ptp(g[lever]) <= TOL or np.ptp(g[gap]) <= TOL or np.ptp(g[risk]) <= TOL:
            continue
        n = len(g)
        r_vg = spearmanr(g[lever], g[gap]).statistic
        r_vr = spearmanr(g[lever], g[risk]).statistic
        r_gr = spearmanr(g[gap], g[risk]).statistic
        ig, ir = int(np.argmin(g[gap].values)), int(np.argmin(g[risk].values))
        rank = lambda s: (s.rank(method="average").values - 1) / (n - 1)
        lv, gv, rv = rank(g[lever]), rank(g[gap]), rank(g[risk])
        rows.append({
            "inst": k, "n": n,
            "rho_v_gap": r_vg, "rho_v_risk": r_vr, "rho_gap_risk": r_gr,
            "opposed": bool(np.sign(r_vg) * np.sign(r_vr) < 0),
            "strong_opposed": bool(np.sign(r_vg) * np.sign(r_vr) < 0
                                   and abs(r_vg) >= 0.5 and abs(r_vr) >= 0.5),
            "exact_reversal": bool(abs(r_vg + 1) < 1e-9 and abs(r_vr - 1) < 1e-9
                                   or abs(r_vg - 1) < 1e-9 and abs(r_vr + 1) < 1e-9),
            "endpoint_opposed": bool(lv[ig] >= 1 - TOL and lv[ir] <= TOL
                                     or lv[ig] <= TOL and lv[ir] >= 1 - TOL),
            "flip": bool(ig != ir),
            "lever_pos_min_gap": float(lv[ig]),
            "lever_pos_min_risk": float(lv[ir]),
            "gap_x_min_gap": float(g[gap].values[ig]),
            "gap_x_min_risk": float(g[gap].values[ir]),
            "risk_x_min_gap": float(g[risk].values[ig]),
            "risk_x_min_risk": float(g[risk].values[ir]),
            "gap_median": float(g[gap].median()),
            "risk_median": float(g[risk].median()),
        })
    return pd.DataFrame(rows)


slot = pd.read_csv(os.path.join(OUT, "llm_slot_candidates.csv"))
slot = slot[slot["gap"].notna()]
bunk = pd.read_csv(os.path.join(OUT, "llm_bunker_candidates.csv"))
bunk = bunk[bunk["gap"].notna()]

fam = {
    "C slot (v = committed TEU)": dict(
        d=slot, inst="inst", gap="gap", risk="overflow", lever="committed_teu"),
    "D bunker (v = lift coverage)": dict(
        d=bunk, inst="inst", gap="gap", risk="overrun", lever="fill"),
}

all_rows, summary = [], []
for name, cfg in fam.items():
    P = per_instance(cfg["d"], cfg["inst"], cfg["gap"], cfg["risk"], cfg["lever"])
    if P.empty:
        print(f"{name}: 无可用实例")
        continue
    P["family"] = name
    all_rows.append(P)
    summary.append({
        "family": name, "instances": len(P),
        "rho(v,gap)": P.rho_v_gap.median(),
        "rho(v,risk)": P.rho_v_risk.median(),
        "rho(gap,risk)": P.rho_gap_risk.median(),
        "opposed": P.opposed.mean(),
        "strong_opposed": P.strong_opposed.mean(),
        "exact_reversal": P.exact_reversal.mean(),
        "EOR": P.endpoint_opposed.mean(),
        "conflict": P.flip.mean(),
        "v_pos(min gap)": P.lever_pos_min_gap.median(),
        "v_pos(min risk)": P.lever_pos_min_risk.median(),
        "risk of min-gap / median": (P.risk_x_min_gap / P.risk_median).median(),
        "gap of min-risk / median": (P.gap_x_min_risk / P.gap_median).median(),
    })

P_all = pd.concat(all_rows, ignore_index=True)
P_all.to_csv(os.path.join(OUT, "prop_instances.csv"), index=False)
S = pd.DataFrame(summary)
S.to_csv(os.path.join(OUT, "prop_table.csv"), index=False)

print("=== 4.2 数值验证：杠杆单调性与端点锁定 ===\n")
print(S.round(3).to_string(index=False))

# 下界检验：conflict 必须 >= EOR >= exact_reversal（命题 1 的直接推论）
print("\n命题 1 的下界检验（每一族都必须成立：conflict >= EOR）")
for _, r in S.iterrows():
    ok = r["conflict"] + 1e-12 >= r["EOR"]
    print(f"  {r['family']}: conflict={r['conflict']:.3f} >= EOR={r['EOR']:.3f}  -> {'OK' if ok else 'VIOLATED'}")

# 条件冲突率：杠杆强对立 vs 其余
print("\n条件冲突率（把'杠杆是否主导排序'拆开看）")
for name in P_all.family.unique():
    p = P_all[P_all.family == name]
    a = p[p.strong_opposed].flip.mean() if p.strong_opposed.any() else np.nan
    b = p[~p.strong_opposed].flip.mean() if (~p.strong_opposed).any() else np.nan
    print(f"  {name}: strong-opposed {a:.3f} (n={int(p.strong_opposed.sum())}) | "
          f"其余 {b:.3f} (n={int((~p.strong_opposed).sum())})")

# 受控合成实验：残余异质性如何削弱端点锁定
rng = np.random.default_rng(2026)
rows = []
for sigma in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30]:
    for rep in range(400):
        n = int(rng.integers(4, 8))
        v = rng.uniform(0, 1, n)
        gap = np.clip(0.55 - 0.35 * v + sigma * rng.standard_normal(n), 0, 1)
        risk = np.maximum(0.0, 0.12 + 0.30 * v + sigma * rng.standard_normal(n))
        rows.append({"sigma": sigma, "flip": int(np.argmin(gap) != np.argmin(risk)),
                     "rho_v_gap": spearmanr(v, gap).statistic,
                     "rho_v_risk": spearmanr(v, risk).statistic})
Z = pd.DataFrame(rows)
G = Z.groupby("sigma").agg(conflict=("flip", "mean"),
                           rho_v_gap=("rho_v_gap", "median"),
                           rho_v_risk=("rho_v_risk", "median")).reset_index()
G.to_csv(os.path.join(OUT, "prop_stress.csv"), index=False)
print("\n=== 受控合成实验：加性异质性 sigma 对冲突率的影响（n=400/档）===")
print(G.round(3).to_string(index=False))

json.dump({"summary": S.to_dict("records"),
           "stress": G.to_dict("records")},
          open(os.path.join(OUT, "prop_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nwrote prop_instances.csv / prop_table.csv / prop_stress.csv / prop_summary.json")
