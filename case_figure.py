#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fig. 2: the lever-spread manipulation on twenty booking books, and the release test.

(a) For every book, the excess exposure of the gap-optimal plan over the
    lowest-risk plan, in two arms that differ only in the spread of committed
    volume: the six full-capacity heuristic plans, and the same six plus three
    deliberate-slack plans (95/90/80 % of capacity).
(b) False-release rate of the gap-only screen and of the two-dimensional screen.
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
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False})

A = pd.read_csv(os.path.join(OUT, "rev2b_books_arms.csv"))
A = A[A.regime == "adverse"]
S = pd.read_csv(os.path.join(OUT, "screen_rule.csv"))

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))

# ---- (a) excess exposure by book, two arms ----
base = A[A.arm == "base"].set_index("book").sort_index()
slack = A[A.arm == "base+slack"].set_index("book").sort_index()
books = sorted(set(base.index) & set(slack.index))
x = np.arange(len(books))
e0 = (base.loc[books, "risk_mingap"] - base.loc[books, "risk_minrisk"]).values
e1 = (slack.loc[books, "risk_mingap"] - slack.loc[books, "risk_minrisk"]).values
ax[0].scatter(x - 0.16, e0, s=19, color="#2c7fb8", zorder=3,
              label="6 full-capacity plans (median spread 0 %)")
ax[0].scatter(x + 0.16, e1, s=19, color="#c0392b", zorder=3,
              label="+ 3 slack plans (median spread 13 %)")
ax[0].axhline(np.median(e0), color="#2c7fb8", lw=0.9, ls="--")
ax[0].axhline(np.median(e1), color="#c0392b", lw=0.9, ls="--")
ax[0].set_xticks(x)
ax[0].set_xticklabels(books, fontsize=6.2)
ax[0].set_xlabel("booking book")
ax[0].set_ylabel("exposure of gap-optimal plan\nminus exposure of safest plan")
ax[0].set_title("(a)  Same books: the penalty scales with spread", fontsize=8.2)
ax[0].legend(fontsize=6.4, frameon=False, loc="center left", bbox_to_anchor=(0.28, 0.32))
ax[0].text(13.2, np.median(e1) + 0.022, f"median {100 * np.median(e1):.1f} pp",
           fontsize=6.4, color="#c0392b", ha="left")

# ---- (b) false release rate ----
q = S[S.gap_cut == 0.50]
labels = ["C slots", "D bunker"]
xi = np.arange(2)
g1 = [q[q.family == "C"].false_rel_gap_only.values[0], q[q.family == "D"].false_rel_gap_only.values[0]]
g2 = [q[q.family == "C"].false_rel_2d.values[0], q[q.family == "D"].false_rel_2d.values[0]]
c1 = [q[q.family == "C"].acc_rate_gap_only.values[0], q[q.family == "D"].acc_rate_gap_only.values[0]]
c2 = [q[q.family == "C"].acc_rate_2d.values[0], q[q.family == "D"].acc_rate_2d.values[0]]
ax[1].bar(xi - 0.18, g1, 0.34, color="#c0392b", label="gap-only screen")
ax[1].bar(xi + 0.18, g2, 0.34, color="#2c7fb8", label="gap + exposure screen")
for i in range(2):
    box = dict(facecolor="white", edgecolor="none", pad=0.5)
    ax[1].text(xi[i] - 0.18, g1[i] + 0.02, f"{g1[i]:.0%}\naccept {c1[i]:.0%}", ha="center", fontsize=6.2, bbox=box)
    ax[1].text(xi[i] + 0.18, g2[i] + 0.02, f"{g2[i]:.0%}\naccept {c2[i]:.0%}", ha="center", fontsize=6.2, bbox=box)
ax[1].axhline(0.25, color="#555555", lw=1.0, ls=":", label="chance (25 %)")
ax[1].set_xticks(xi)
ax[1].set_xticklabels(labels)
ax[1].set_ylim(0, 0.72)
ax[1].set_ylabel("false release rate")
ax[1].set_title("(b)  The exposure test removes the excess", fontsize=8.2)
ax[1].legend(fontsize=6.4, frameon=False, loc="upper right")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_screening.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_screening.pdf"), bbox_inches="tight")
print("saved fig2_screening.png / .pdf")
print("base excess median %.4f  slack excess median %.4f  ratio %.1f"
      % (np.median(e0), np.median(e1), np.median(e1) / max(1e-9, np.median(e0))))
