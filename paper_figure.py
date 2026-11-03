#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单张四面板主图（Fig. 1）：把 A/B（修复制 vs 承诺制，同一批 LLM 解）与 C/D（真 LLM 候选）合并。

panel (a)  rho(gap, risk)：比率口径 vs 绝对口径
panel (b)  top-1 冲突率：gap 选出的方案不是风险最低的方案
panel (c)  端点位置：风险最低方案的 gap 位置 / gap 最低方案的风险位置
panel (d)  机制：承诺量杠杆的实例内排名 → gap 排名（蓝）与风险排名（红），C/D 两族
输出 fig1_combined.png / fig1_combined.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8.4, "axes.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False})

T = pd.read_csv(os.path.join(OUT, "unified_table_llm.csv"))
# 列名是中文，按顺序取，避免字体问题
cols = list(T.columns)
fam = ["A", "B", "C", "D"]
label = ["A  repair\n(regret)", "B  commit\n(violation)", "C  slots\n(overflow)", "D  bunker\n(overrun)"]
rho_r = T["rho_比率"].values
rho_a = T["rho_绝对"].values
conf = T["top1反转"].values
pos_safe = T["最稳方案gap位置"].values
pos_mingap = T["min-gap风险位置"].values
n_inst = T["实例数"].values

fig = plt.figure(figsize=(6.4, 6.1))
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.27)

# ---------- (a) rho under two normalisations ----------
ax = fig.add_subplot(gs[0, 0])
x = np.arange(4)
ax.bar(x - 0.19, rho_r, 0.36, color="#c0392b", label="ratio-normalised")
ax.bar(x + 0.19, rho_a, 0.36, color="#2c7fb8", label="absolute")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(label, fontsize=7.2)
ax.set_ylabel(r"median Spearman $\rho$(gap, risk)")
ax.set_ylim(-1.15, 1.15)
ax.set_title("(a)  Direction depends on normalisation")
for xi, vr, va in zip(x, rho_r, rho_a):
    if abs(vr - va) < 0.03:
        # the two normalisations agree: one label above the pair
        ax.text(xi, max(vr, va) + (0.06 if vr >= 0 else -0.13), f"{vr:+.2f}",
                ha="center", fontsize=6.6)
    else:
        ax.text(xi - 0.19, vr + (0.06 if vr >= 0 else -0.13), f"{vr:+.2f}",
                ha="center", fontsize=6.6)
        ax.text(xi + 0.19, va + (0.06 if va >= 0 else -0.13), f"{va:+.2f}",
                ha="center", fontsize=6.6)
ax.legend(loc="lower left", fontsize=6.8, frameon=False)

# ---------- (b) top-1 conflict ----------
ax = fig.add_subplot(gs[0, 1])
colors = ["#7fb3d5", "#c0392b", "#c0392b", "#c0392b"]
ax.bar(x, conf, 0.55, color=colors)
ax.axhline(0.75, ls="--", lw=0.8, color="gray")
ax.set_xticks(x)
ax.set_xticklabels(label, fontsize=7.2)
ax.set_ylim(0, 1.06)
ax.set_ylabel("top-1 conflict rate")
ax.set_title("(b)  The gap's pick is the wrong plan")
for xi, v, n in zip(x, conf, n_inst):
    ax.text(xi, v + 0.03, f"{v:.0%}\n(n={int(n)})", ha="center", fontsize=7)

# ---------- (c) endpoint positions ----------
ax = fig.add_subplot(gs[1, 0])
ax.barh(x + 0.19, pos_safe, 0.34, color="#c0392b", label="gap rank of the safest plan")
ax.barh(x - 0.19, pos_mingap, 0.34, color="#2c7fb8", label="risk rank of the lowest-gap plan")
ax.set_yticks(x)
ax.set_yticklabels(label, fontsize=7.2)
ax.set_xlim(0, 1.14)
ax.axvline(1.0, ls="--", lw=0.8, color="gray")
ax.set_xlabel("within-instance rank  (0 = lowest, 1 = highest)")
ax.set_title("(c)  Two minima, opposite ends of the pool")
for yi, a, b in zip(x, pos_safe, pos_mingap):
    ax.text(a + 0.02, yi + 0.19, f"{a:.2f}", va="center", fontsize=6.6)
    ax.text(b + 0.02, yi - 0.19, f"{b:.2f}", va="center", fontsize=6.6)
ax.legend(fontsize=6.6, loc="lower right", frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.94, borderpad=0.25)

# ---------- (d) mechanism: lever rank vs metric rank ----------
ax = fig.add_subplot(gs[1, 1])


def rank_within(df, inst, col):
    return df.groupby(inst)[col].rank(method="average", pct=True) - 0.5 / df.groupby(inst)[col].transform("size")


slot = pd.read_csv(os.path.join(OUT, "llm_slot_candidates.csv"))
slot = slot[slot.gap.notna()]
bunk = pd.read_csv(os.path.join(OUT, "llm_bunker_candidates.csv"))
bunk = bunk[bunk.gap.notna()]

series = []
if "committed_teu" in slot.columns:
    s = pd.DataFrame({"v": rank_within(slot, "inst", "committed_teu"),
                      "gap": rank_within(slot, "inst", "gap"),
                      "risk": rank_within(slot, "inst", "overflow")})
    series += [("C: gap", s.v, s.gap, "#1f6fb4", "o"), ("C: risk", s.v, s.risk, "#e07b39", "o")]
if "fill" in bunk.columns:
    b = pd.DataFrame({"v": rank_within(bunk, "inst", "fill"),
                      "gap": rank_within(bunk, "inst", "gap"),
                      "risk": rank_within(bunk, "inst", "overrun")})
    series += [("D: gap", b.v, b.gap, "#2c7fb8", "^"), ("D: risk", b.v, b.risk, "#c0392b", "^")]

for nm, vx, vy, c, mk in series:
    ax.scatter(vx, vy, s=9, alpha=0.28, color=c, marker=mk, edgecolors="none")
    # 分箱中位数趋势线
    bins = np.clip((vx * 8).astype(int), 0, 7)
    med = pd.DataFrame({"b": bins, "y": vy}).groupby("b").y.median()
    ax.plot((med.index + 0.5) / 8, med.values, color=c, lw=2.0,
            marker=mk, ms=5, label=nm, ls="-" if "gap" in nm else "--")

ax.set_xlabel("within-instance rank of the commitment lever  v")
ax.set_ylabel("within-instance rank of the metric")
ax.set_title("(d)  Opposed along the same lever")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=6.4, frameon=False, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.16))

fig.savefig(os.path.join(OUT, "fig1_combined.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_combined.pdf"), bbox_inches="tight")
print("saved fig1_combined.png / fig1_combined.pdf")
