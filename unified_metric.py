#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一口径总表：gap 与真实风险的相关性，在"比率口径"与"绝对口径"下是否一致。

机制（已由四族数据检验，不再使用"分母 CV 驱动 Δρ"的旧假设）：
  gap 与风险都单调依赖同一个决策杠杆——承诺量（接受订舱量 / 加油量），但方向相反：
  承诺越多，gap 越小、漂移下风险越大。因此两族的最小值必然落在候选集的两个端点，
  top-1 排序冲突是结构性的；比率口径的失真只在此基础上做二阶放大。

四组数据：
  A  LLM 解 + 对偶间隙 -> 成本后悔（分母 = 实例内最优目标，常数）
  B  LLM 解 + 承诺制   -> 0/1 违约率（分母 = 情景数，常数）
  C  海运舱位分配       -> 甩柜量（分母 = 该方案的承诺箱量，随方案变）
  D  海运加油港选择     -> 超支额（分母 = 该方案自身成本，随方案变）
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
USE_LLM = "--llm" in sys.argv       # C/D 换成真 LLM 候选口径
SUF = "_llm" if USE_LLM else ""


def per_instance(df, inst_col, gap_col, ratio_col, abs_col, denom_cv=None):
    rows = []
    for inst, g in df.groupby(inst_col):
        g = g.dropna(subset=[gap_col, ratio_col])
        # 风险在实例内不变化时，argmin 只是一次任意平局，排序冲突无从谈起 → 剔除
        if len(g) < 3 or np.ptp(g[gap_col]) == 0 or np.ptp(g[ratio_col]) == 0:
            continue
        r1 = spearmanr(g[gap_col], g[ratio_col]).statistic
        if abs_col is not None and g[abs_col].notna().all() and np.ptp(g[abs_col]) > 0:
            r2 = spearmanr(g[gap_col], g[abs_col]).statistic
        else:
            r2 = r1
        ig, iv = int(np.argmin(g[gap_col].values)), int(np.argmin(g[ratio_col].values))
        rows.append({"inst": inst, "rho_ratio": r1, "rho_abs": r2,
                     "flip": ig != iv, "n": len(g),
                     "cv_denom": denom_cv(g) if denom_cv else np.nan})
    return pd.DataFrame(rows)


def cv_of_ratio(df, ratio_col, abs_col):
    d = df[(df[ratio_col] > 1e-9) & (df[abs_col].notna())]
    return (d[abs_col] / d[ratio_col]).std() / (d[abs_col] / d[ratio_col]).mean()


def extremes(df, inst_col, gap_col, risk_col):
    """端点检验（0=实例内最小，1=最大）：
       gap_pos_min_risk  : 风险最小方案的 gap 位置（越接近 1，越说明"最稳"=最差方案）
       risk_pos_min_gap  : gap 最小方案的风险位置（越接近 1，越说明"最好"=最险方案）"""
    rows = []
    for inst, g in df.groupby(inst_col):
        g = g.dropna(subset=[gap_col, risk_col])
        if len(g) < 3 or np.ptp(g[gap_col]) == 0 or np.ptp(g[risk_col]) == 0:
            continue
        n = len(g)
        gp = (g[gap_col].rank(method="average").values - 1) / (n - 1)
        rp = (g[risk_col].rank(method="average").values - 1) / (n - 1)
        ig, iv = int(np.argmin(g[gap_col].values)), int(np.argmin(g[risk_col].values))
        rows.append({"inst": inst, "n": n, "opposed": bool(gp[iv] >= 1 - 1e-9),
                     "gap_pos_min_risk": float(gp[iv]), "risk_pos_min_gap": float(rp[ig])})
    return pd.DataFrame(rows)


sets, pairs = [], []

# A：成本后悔（分母 = 实例常数）
a = pd.read_csv(os.path.join(OUT, "regret_pilot_candidates_adverse.csv"))
a["abs_proxy"] = a["mean_regret"] * a["objective"].abs()
ra = per_instance(a, "instance", "gap", "mean_regret", "abs_proxy")
sets.append(("A regret(inst) 成本后悔（实例常数）", ra, 0.0))
pairs.append(("A regret(inst) 成本后悔（实例常数）", a, "instance", "gap", "mean_regret"))

# B：0/1 违约率（分母 = 情景数，常数）
b = pd.read_csv(os.path.join(OUT, "audit2_candidates.csv"))
b = b[b["viol_rate"].notna()]
rb = per_instance(b, "instance", "gap", "viol_rate", None)
sets.append(("B violation 0/1（情景数）", rb, 0.0))
pairs.append(("B violation 0/1（情景数）", b, "instance", "gap", "viol_rate"))

# C：舱位分配甩柜
c = pd.read_csv(os.path.join(OUT, f"llm_slot_candidates.csv" if USE_LLM else "sea_slot_candidates.csv"))
rc = per_instance(c, "inst", "gap", "overflow", "abs_overflow", denom_cv=None)
cv_c = np.nanmedian([cv_of_ratio(g, "overflow", "abs_overflow") for _, g in c.groupby("inst")])
sets.append((("C slot overflow（LLM，承诺箱量）" if USE_LLM else "C slot overflow（承诺箱量）"), rc, cv_c))
pairs.append((("C slot overflow（LLM，承诺箱量）" if USE_LLM else "C slot overflow（承诺箱量）"),
              c, "inst", "gap", "overflow"))

# D：加油港超支
d = pd.read_csv(os.path.join(OUT, f"llm_bunker_candidates.csv" if USE_LLM else "sea_bunker_candidates.csv"))
rd = per_instance(d, "inst", "gap", "overrun", "abs_overrun", denom_cv=None)
cv_d = np.nanmedian([cv_of_ratio(g, "overrun", "abs_overrun") for _, g in d.groupby("inst")])
sets.append((("D bunker overrun（LLM，自身成本）" if USE_LLM else "D bunker overrun（自身成本）"), rd, cv_d))
pairs.append((("D bunker overrun（LLM，自身成本）" if USE_LLM else "D bunker overrun（自身成本）"),
              d, "inst", "gap", "overrun"))

tab = []
for name, r, cv in sets:
    if r.empty:
        continue
    ex = next((p for p in pairs if p[0] == name), None)
    ex = extremes(ex[1], ex[2], ex[3], ex[4]) if ex else pd.DataFrame()
    tab.append({"度量": name, "实例数": len(r), "CV(分母)": cv,
                "rho_比率": r.rho_ratio.median(), "rho_绝对": r.rho_abs.median(),
                "Delta_rho": abs(r.rho_ratio.median() - r.rho_abs.median()),
                "top1反转": r.flip.mean(),
                "端点对立率": ex.opposed.mean() if len(ex) else np.nan,
                "最稳方案gap位置": ex.gap_pos_min_risk.median() if len(ex) else np.nan,
                "min-gap风险位置": ex.risk_pos_min_gap.median() if len(ex) else np.nan})
T = pd.DataFrame(tab)
T.to_csv(os.path.join(OUT, f"unified_table{SUF}.csv"), index=False)
print(T.round(3).to_string(index=False))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2))
    y = np.arange(len(T))
    short = [s.split(" ")[0] for s in T["度量"]]
    ax[0].barh(y - 0.18, T["rho_比率"], height=0.34, label="ratio-normalised", color="#c0392b")
    ax[0].barh(y + 0.18, T["rho_绝对"], height=0.34, label="absolute", color="#2c7fb8")
    ax[0].set_yticks(y); ax[0].set_yticklabels(short, fontsize=10)
    ax[0].axvline(0, color="k", lw=0.8); ax[0].set_xlabel("median Spearman rho (gap vs risk)")
    ax[0].legend(fontsize=8); ax[0].set_title("rho(gap, risk) under two normalisations", fontsize=10)

    colors = ["#2c7fb8", "#c0392b", "#c0392b", "#c0392b"]
    ax[1].bar(y, T["top1反转"], color=colors, width=0.55)
    ax[1].set_xticks(y); ax[1].set_xticklabels(short, fontsize=10)
    ax[1].set_ylim(0, 1.0); ax[1].set_ylabel("top-1 flip rate")
    ax[1].set_title("Gap's top pick is the wrong plan", fontsize=10)
    ax[1].axhline(0.2, ls="--", lw=0.8, color="gray")
    for i, v in enumerate(T["top1反转"]):
        ax[1].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)

    ax[2].barh(y - 0.18, T["最稳方案gap位置"], height=0.34, label="gap rank of safest plan", color="#c0392b")
    ax[2].barh(y + 0.18, T["min-gap风险位置"], height=0.34, label="risk rank of best-gap plan", color="#2c7fb8")
    ax[2].set_yticks(y); ax[2].set_yticklabels(short, fontsize=10)
    ax[2].set_xlim(0, 1.05); ax[2].set_xlabel("within-instance rank (0 = lowest, 1 = highest)")
    ax[2].axvline(1.0, ls="--", lw=0.8, color="gray")
    ax[2].legend(fontsize=8); ax[2].set_title("Risk metric rewards inaction", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"unified_fig{SUF}.png"), dpi=160)
    print(f"saved unified_fig{SUF}.png")
except Exception as e:
    print("plot skipped:", e)

# 候选生成器稳健性：启发式 vs 真 LLM（C、D 两族）
if USE_LLM:
    def _rows(fn, inst, gp, rt, ab):
        x = pd.read_csv(os.path.join(OUT, fn))
        r = per_instance(x, inst, gp, rt, ab)
        return r if not r.empty else None
    rob = []
    for tag, fam in [("C 舱位分配", "slot"), ("D 加油港", "bunker")]:
        if fam == "slot":
            h = _rows("sea_slot_candidates.csv", "inst", "gap", "overflow", "abs_overflow")
            l = _rows("llm_slot_candidates.csv", "inst", "gap", "overflow", "abs_overflow")
        else:
            h = _rows("sea_bunker_candidates.csv", "inst", "gap", "overrun", "abs_overrun")
            l = _rows("llm_bunker_candidates.csv", "inst", "gap", "overrun", "abs_overrun")
        for nm, r in [("heuristic", h), ("llm", l)]:
            if r is None:
                continue
            rob.append({"族": tag, "候选": nm, "实例数": len(r),
                        "rho_比率": r.rho_ratio.median(), "rho_绝对": r.rho_abs.median(),
                        "Delta_rho": abs(r.rho_ratio.median() - r.rho_abs.median()),
                        "top1反转": r.flip.mean()})
    R = pd.DataFrame(rob)
    R.to_csv(os.path.join(OUT, "unified_robust.csv"), index=False)
    print("\n候选生成器稳健性")
    print(R.round(3).to_string(index=False))
