#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
审稿补充实验（对应 referee report 的 R2/R3/R6 与 T1/T5）：
  1. 证书序不变性：把「真最优 gap」换成「LP 对偶界 gap」，检验排序是否改变（应完全相同）。
  2. 统计推断：Wilson 95% CI + 实例级 bootstrap；并报告不过滤（平局计为一致）口径。
  3. 效应量：甩柜率 / 超支金额层面的「安全代价」，不只秩冲突。
  4. 杠杆敏感性：替代杠杆 + 配对分层，检验冲突是否只由单一 v 驱动。
  5. 二维验收规则（gap + 暴露度）的假放行率。
输出：certificate_check.csv / stats_ci.csv / effect_size.csv / lever_pairs.csv / screen_rule.csv
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))


def load_module(fn, stop="inst_rows, cand_rows"):
    src = open(os.path.join(OUT, fn), encoding="utf-8").read().split(stop)[0]
    src = src.replace('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = "."')
    g = {"__name__": "lib"}
    exec(src, g)
    return g


slot = load_module("sea_slot_pilot.py")
bunk = load_module("sea_bunker_pilot.py")

# ---------------- 1. 证书序不变性 ----------------
rows = []
for s in range(40):
    inst = slot["gen_slot"](1000 + s, n_book=int(18 + 6 * (s % 3)), cap=8000,
                            tight=[1.05, 1.15, 1.30][s % 3])
    opt, _ = slot["solve"](inst)
    lp = linprog(c=-inst["r"], A_ub=inst["A"], b_ub=inst["b"],
                 bounds=[(0, 1)] * inst["n"], method="highs")
    if not lp.success:
        continue
    lpopt = -lp.fun
    rows.append({"family": "C", "inst": 1000 + s, "opt": opt, "bound": lpopt,
                 "bound_ratio": lpopt / opt})
C = pd.DataFrame(rows)

rows = []
for s in range(40):
    inst = bunk["gen"](2000 + s)
    opt, _, _ = bunk["solve"](inst)
    M = len(inst["p"])
    f = np.array([bunk["fixed_cost"](inst, j) for j in range(M)])
    A = np.hstack([np.eye(M), -np.diag(inst["cap"])])
    A = np.vstack([A, np.concatenate([-np.ones(M), np.zeros(M)])])
    b = np.concatenate([np.zeros(M), [-inst["B"]]])
    lp = linprog(c=np.concatenate([inst["p"], f]), A_ub=A, b_ub=b,
                 bounds=[(0, cap) for cap in inst["cap"]] + [(0, 1)] * M, method="highs")
    if not lp.success:
        continue
    rows.append({"family": "D", "inst": 2000 + s, "opt": opt, "bound": lp.fun,
                 "bound_ratio": lp.fun / opt})
D = pd.DataFrame(rows)
BND = pd.concat([C, D])

cert, mag = [], []
for fam, fn, gcol, icol in [("C", "llm_slot_candidates.csv", "gap", "inst"),
                            ("D", "llm_bunker_candidates.csv", "gap", "inst")]:
    df = pd.read_csv(os.path.join(OUT, fn))
    df = df[df[gcol].notna()]
    bd = BND[BND.family == fam].set_index("inst")
    recs = []
    for i, g in df.groupby(icol):
        if i not in bd.index or len(g) < 3:
            continue
        bound = float(bd.loc[i, "bound"])
        objv = g["obj"].values if fam == "C" else g["cost"].values
        # C 为最大化：g = (bound − obj)/|bound|；D 为最小化：g = (cost − bound)/|bound|
        g2 = (bound - objv) / abs(bound) if fam == "C" else (objv - bound) / abs(bound)
        g1 = g[gcol].values
        rho = spearmanr(g1, g2).statistic
        # 排序是否完全一致 + argmin 是否同一方案
        same = bool(int(np.argmin(g1)) == int(np.argmin(g2)))
        risk = g["overflow"].values if fam == "C" else g["overrun"].values
        recs.append({"family": fam, "inst": i,
                     "rho_true_vs_bound": rho, "same_argmin": same,
                     "conflict_true": bool(int(np.argmin(g1)) != int(np.argmin(risk))),
                     "conflict_bound": bool(int(np.argmin(g2)) != int(np.argmin(risk))),
                     "acc5_true": float(np.mean(g1 <= 0.05)),
                     "acc5_bound": float(np.mean(g2 <= 0.05)),
                     "gap_true_med": float(np.median(g1)), "gap_bound_med": float(np.median(g2))})
    R = pd.DataFrame(recs)
    cert.append(R)
    mag.append({"family": fam, "bound_ratio_med": float(bd.bound_ratio.median()),
                "rho_med": float(R.rho_true_vs_bound.median()),
                "same_argmin_rate": float(R.same_argmin.mean()),
                "conflict_true": float(R.conflict_true.mean()),
                "conflict_bound": float(R.conflict_bound.mean()),
                "acc5_true": float(R.acc5_true.mean()), "acc5_bound": float(R.acc5_bound.mean()),
                "gap_true_med": float(R.gap_true_med.median()),
                "gap_bound_med": float(R.gap_bound_med.median())})
CERT = pd.concat(cert)
CERT.to_csv(os.path.join(OUT, "certificate_check.csv"), index=False)
CERTM = pd.DataFrame(mag)
CERTM.to_csv(os.path.join(OUT, "certificate_magnitude.csv"), index=False)
print("=== 1. 证书序不变性 ===")
print(CERTM.round(3).to_string(index=False))

# ---------------- 2. CI + 不过滤口径 ----------------
def perfam(df, inst, gap, risk):
    out = []
    for i, g in df.groupby(inst):
        g = g.dropna(subset=[gap, risk])
        out.append({"inst": i, "n": len(g),
                    "conflict": bool(int(np.argmin(g[gap].values)) != int(np.argmin(g[risk].values))),
                    "var": float(np.ptp(g[risk].values)),
                    "rho": spearmanr(g[gap], g[risk]).statistic if len(g) >= 3 and np.ptp(g[risk]) else np.nan})
    return pd.DataFrame(out)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


a = pd.read_csv(os.path.join(OUT, "regret_pilot_candidates_adverse.csv"))
a["key"] = a.problem_type.astype(str) + "|" + a.instance.astype(str)
b = pd.read_csv(os.path.join(OUT, "audit2_candidates.csv"))
b["key"] = b.problem_type.astype(str) + "|" + b.instance.astype(str)
c = pd.read_csv(os.path.join(OUT, "llm_slot_candidates.csv"))
c = c[c.feasible == True]
d = pd.read_csv(os.path.join(OUT, "llm_bunker_candidates.csv"))

FAM = [("A", perfam(a, "key", "gap", "mean_regret")),
       ("B", perfam(b, "key", "gap", "viol_rate")),
       ("C", perfam(c, "inst", "gap", "overflow")),
       ("D", perfam(d, "inst", "gap", "overrun"))]

rng = np.random.default_rng(20261028)
stat_rows = []
for nm, P in FAM:
    adv = P[P.n >= 2]
    fil = adv[(adv.n >= 3) & (adv["var"] > 0)]
    r = fil.conflict.values
    boot = [rng.choice(r, len(r)).mean() for _ in range(2000)]
    lo, hi = wilson(int(r.sum()), len(r))
    stat_rows.append({"family": nm, "n_all": len(adv), "conflict_all": adv.conflict.mean(),
                      "n_filtered": len(fil), "conflict_filtered": r.mean(),
                      "ci_lo": lo, "ci_hi": hi,
                      "boot_lo": np.percentile(boot, 2.5), "boot_hi": np.percentile(boot, 97.5),
                      "rho_med": fil.rho.median()})
STAT = pd.DataFrame(stat_rows)
STAT.to_csv(os.path.join(OUT, "stats_ci.csv"), index=False)
print("\n=== 2. 冲突率：不过滤 / 过滤 + 95% CI ===")
print(STAT.round(3).to_string(index=False))

# ---------------- 3. 效应量 ----------------
eff = []
for nm, df, inst, gap, risk, rk, scale in [
        ("C slots", c, "inst", "gap", "overflow", "overflow", 1.0),
        ("D bunker", d, "inst", "gap", "overrun", "overrun", 1.0)]:
    recs = []
    for i, g in df.groupby(inst):
        g = g.dropna(subset=[gap, risk])
        if len(g) < 3 or np.ptp(g[risk]) == 0:
            continue
        ig, ir = int(np.argmin(g[gap].values)), int(np.argmin(g[risk].values))
        recs.append({"risk_mingap": g[risk].values[ig], "risk_minrisk": g[risk].values[ir],
                     "risk_med": float(np.median(g[risk].values)),
                     "gap_mingap": g[gap].values[ig], "gap_minrisk": g[gap].values[ir],
                     "riskrank_mingap": float((g[risk].rank().values - 1)[ig] / (len(g) - 1))})
    R = pd.DataFrame(recs)
    eff.append({"family": nm, "n": len(R),
                "risk_mingap_med": R.risk_mingap.median(),
                "risk_minrisk_med": R.risk_minrisk.median(),
                "risk_med_med": R.risk_med.median(),
                "rel_reduction": 1 - R.risk_minrisk.median() / max(1e-12, R.risk_mingap.median()),
                "gap_mingap_med": R.gap_mingap.median(),
                "gap_minrisk_med": R.gap_minrisk.median(),
                "share_mingap_is_worst": float((R.riskrank_mingap >= 1 - 1e-9).mean()),
                "share_mingap_top_quartile": float((R.riskrank_mingap >= 0.75 - 1e-9).mean())})
EFF = pd.DataFrame(eff)
EFF.to_csv(os.path.join(OUT, "effect_size.csv"), index=False)
print("\n=== 3. 效应量（安全代价） ===")
print(EFF.round(3).to_string(index=False))

# 金额口径（D）
db = d.dropna(subset=["gap", "abs_overrun"])
recs = []
for i, g in db.groupby("inst"):
    if len(g) < 3 or np.ptp(g.abs_overrun) == 0:
        continue
    ig, ir = int(np.argmin(g.gap.values)), int(np.argmin(g.abs_overrun.values))
    recs.append((g.abs_overrun.values[ig], g.abs_overrun.values[ir]))
recs = np.array(recs)
print(f"D 绝对超支（USD/inst）：min-gap 中位 {np.median(recs[:,0]):,.0f} | "
      f"min-risk 中位 {np.median(recs[:,1]):,.0f} | 相对削减 "
      f"{1-np.median(recs[:,1])/np.median(recs[:,0]):.3f}")

# ---------------- 4. 杠杆敏感性 ----------------
def lever_table(df, fam, inst, gap, risk, levers):
    out = []
    for nm, col, direction in levers:
        rr, conf = [], []
        for i, g in df.groupby(inst):
            g = g.dropna(subset=[gap, risk, col])
            if len(g) < 3 or np.ptp(g[risk]) == 0 or np.ptp(g[col]) == 0:
                continue
            r1 = spearmanr(g[col], g[gap]).statistic
            r2 = spearmanr(g[col], g[risk]).statistic
            rr.append(np.sign(r1) * np.sign(r2))
            conf.append(int(np.argmin(g[gap].values)) != int(np.argmin(g[risk].values)))
        out.append({"family": fam, "lever": nm,
                    "n": len(rr), "opposed": float(np.mean(np.array(rr) < 0)),
                    "conflict": float(np.mean(conf))})
    return out


lev_c = lever_table(c, "C", "inst", "gap", "overflow",
                    [("committed TEU", "committed_teu", 1), ("revenue", "obj", 1),
                     ("bookings accepted", "n_acc", 1), ("peak utilization", "util", 1)])
lev_d = lever_table(d, "D", "inst", "gap", "overrun",
                    [("lift coverage", "fill", 1), ("lift tonnes", "lift_t", 1),
                     ("ports called", "n_ports", 1), ("total cost", "cost", 1)])
LEV = pd.DataFrame(lev_c + lev_d)
LEV.to_csv(os.path.join(OUT, "lever_alt.csv"), index=False)
print("\n=== 4a. 替代杠杆 ===")
print(LEV.round(3).to_string(index=False))

# 配对分层：|Δv| 小 vs 大时，gap 与 risk 是否仍反向
pairs = []
for nm, df, inst, gap, risk, col, sign in [
        ("C", c, "inst", "gap", "overflow", "committed_teu", 1),
        ("D", d, "inst", "gap", "overrun", "fill", 1)]:
    for i, g in df.groupby(inst):
        g = g.dropna(subset=[gap, risk, col])
        if len(g) < 3 or np.ptp(g[risk]) == 0 or np.ptp(g[col]) == 0:
            continue
        v = (g[col].rank().values - 1) / (len(g) - 1)
        gp, rp = g[gap].values, g[risk].values
        for a1 in range(len(g)):
            for b1 in range(a1 + 1, len(g)):
                if abs(gp[a1] - gp[b1]) < 1e-12 or abs(rp[a1] - rp[b1]) < 1e-12:
                    continue
                pairs.append({"family": nm, "dv": abs(v[a1] - v[b1]),
                              "discordant": int(np.sign(gp[a1] - gp[b1]) != np.sign(rp[a1] - rp[b1]))})
P = pd.DataFrame(pairs)
P["dv_bin"] = pd.qcut(P.dv, 3, labels=["small", "mid", "large"])
PAIR = P.groupby(["family", "dv_bin"]).agg(n=("discordant", "size"),
                                           discord=("discordant", "mean")).reset_index()
PAIR.to_csv(os.path.join(OUT, "lever_pairs.csv"), index=False)
print("\n=== 4b. 配对分层（|Δv| 越大越冲突 → 说明冲突由 v 驱动；若小 Δv 仍冲突，则存在非杠杆结构） ===")
print(PAIR.round(3).to_string(index=False))

# ---------------- 5. 二维验收规则 ----------------
PLANS = []


def rule_table(nm, df, inst, gap, risk, expo_col, expo_dir, cut=0.25):
    recs = []
    for i, g in df.groupby(inst):
        g = g.dropna(subset=[gap, risk, expo_col])
        if len(g) < 3 or np.ptp(g[risk]) == 0:
            continue
        n = len(g)
        rk = (g[risk].rank().values - 1) / (n - 1)
        gk = (g[gap].rank().values - 1) / (n - 1)      # 实例内 gap 分位（越小越好）
        er = (g[expo_col].rank().values - 1) / (n - 1)
        if expo_dir < 0:
            er = 1 - er
        for idx in range(n):
            recs.append({"family": nm, "gap_rank": gk[idx], "risk_rank": rk[idx],
                         "expo_rank": er[idx]})
    R = pd.DataFrame(recs)
    PLANS.append(R)
    out = []
    for cut in (0.25, 0.50):
        acc = R[R.gap_rank <= cut + 1e-9]
        acc2 = acc[acc.expo_rank <= 0.5 + 1e-9]
        out.append({"family": nm, "gap_cut": cut,
                    "acc_rate_gap_only": len(acc) / len(R),
                    "false_rel_gap_only": float((acc.risk_rank >= 0.75 - 1e-9).mean()) if len(acc) else np.nan,
                    "acc_rate_2d": len(acc2) / len(R),
                    "false_rel_2d": float((acc2.risk_rank >= 0.75 - 1e-9).mean()) if len(acc2) else np.nan,
                    "mean_riskrank_gap_only": float(acc.risk_rank.mean()) if len(acc) else np.nan,
                    "mean_riskrank_2d": float(acc2.risk_rank.mean()) if len(acc2) else np.nan,
                    "n_plans": len(R)})
    return out


SCR = pd.DataFrame(rule_table("C", c, "inst", "gap", "overflow", "committed_teu", 1) +
                   rule_table("D", d, "inst", "gap", "overrun", "fill", -1))
SCR.to_csv(os.path.join(OUT, "screen_rule.csv"), index=False)
pd.concat(PLANS, ignore_index=True).to_csv(os.path.join(OUT, "screen_plans.csv"), index=False)
print("\n=== 5. 验收规则：gap-only vs gap+暴露度（假放行 = 被接受且风险位于前 25%） ===")
print(SCR.round(3).to_string(index=False))

json.dump({"certificate": CERTM.round(4).to_dict("records"),
           "stats": STAT.round(4).to_dict("records"),
           "effect": EFF.round(4).to_dict("records"),
           "screen": SCR.round(4).to_dict("records")},
          open(os.path.join(OUT, "rigor_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=float)
print("\nsaved certificate_check.csv / stats_ci.csv / effect_size.csv / lever_alt.csv / lever_pairs.csv / screen_rule.csv")
