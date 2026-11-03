#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-render Fig. 3 at submission resolution from the archived candidate file.

The plotting code mirrors real_case.py; it reads real_case_plans.csv instead of
re-solving, so the published numbers are untouched. Output: Figure_3.png (300 dpi)
and Figure_3.pdf (vector).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False})

C = pd.read_csv(os.path.join(OUT, "real_case_plans.csv"))
ig = C.gap.idxmin()
ir = C.risk.idxmin()

fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.05))
style = {"llm": ("#d35400", "X", 34), "slack": ("#7f8c8d", "D", 26), "heur": ("#2c7fb8", "o", 24)}
for _, r in C.iterrows():
    key = "llm" if r.plan.startswith("llm") else ("slack" if r.plan.startswith("slack") else "heur")
    c_, m_, s_ = style[key]
    ax[0].scatter(r.gap, r.risk, c=c_, marker=m_, s=s_, alpha=0.9, zorder=3)
ax[0].scatter([C.loc[ig, "gap"]], [C.loc[ig, "risk"]], s=120, facecolors="none",
              edgecolors="k", lw=1.5, zorder=4)
ax[0].scatter([C.loc[ir, "gap"]], [C.loc[ir, "risk"]], s=120, marker="s", facecolors="none",
              edgecolors="k", lw=1.5, zorder=4)
ax[0].annotate("gap's top pick\n(fills the ship)", xy=(C.loc[ig, "gap"], C.loc[ig, "risk"]),
               xytext=(0.10, 0.235), fontsize=6.9, arrowprops=dict(arrowstyle="->", lw=0.8))
ax[0].annotate("lowest risk\n(carries almost nothing)", xy=(C.loc[ir, "gap"], C.loc[ir, "risk"]),
               xytext=(0.32, 0.055), fontsize=6.9, arrowprops=dict(arrowstyle="->", lw=0.8))
ax[0].set_xlabel("optimality gap of the plan")
ax[0].set_ylabel("realised overrun of the committed contribution")
ax[0].set_title(f"One voyage, {len(C)} candidate plans", fontsize=8.4)
ax[0].set_xlim(-0.03, 0.83)
ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color="#2c7fb8", label="heuristic"),
                      Line2D([], [], marker="X", ls="", color="#d35400", label="LLM"),
                      Line2D([], [], marker="D", ls="", color="#7f8c8d", label="deliberate slack"),
                      Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=6, label="min gap"),
                      Line2D([], [], marker="s", ls="", mfc="none", mec="k", ms=6, label="min risk")],
             fontsize=6.5, frameon=False, loc="upper right")

ax2 = ax[1].twinx()
ax[1].scatter(C.teu, C.gap, c="#2c7fb8", marker="o", s=24, label="gap (left axis)")
ax2.scatter(C.teu, C.risk, c="#c0392b", marker="^", s=24, label="realised risk (right axis)")
ax[1].set_xlabel("committed TEU on the voyage")
ax[1].set_ylabel("optimality gap", color="#2c7fb8")
ax2.set_ylabel("realised risk", color="#c0392b")
ax2.spines["top"].set_visible(False)
ax[1].set_title("Opposite monotonicity in the commitment lever", fontsize=8.4)
ax[1].tick_params(axis="y", colors="#2c7fb8")
ax2.tick_params(axis="y", colors="#c0392b")
h1, l1 = ax[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper center")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "Figure_3.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "Figure_3.pdf"), bbox_inches="tight")
print("saved Figure_3.png / Figure_3.pdf")
