#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figures 4-10 for the TR-C submission.

Every panel is computed from an archived CSV in this directory. Style follows
the reference layout: Times/STIX serif 9 pt, spines top/right off, #2c7fb8 blue /
#c0392b red. Two figures are three-dimensional: Fig. 5 (a genuine third axis,
committed TEU) and Fig. 8(a) (penalty over the family x regime grid).

Figure_4  the magnitude law of Corollary 1
Figure_5  the nineteen voyage plans in (log gap, exposure, committed TEU)
Figure_6  the lever mechanism in one instance per maritime family
Figure_7  the nine effect sizes with Wilson and bootstrap intervals
Figure_8  drift x family: penalty surface and conflict rate
Figure_9  release-rule operating points against the chance baseline
Figure_10 the resolution limit: gap spread against exposure spread
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300,
})
BLUE, RED, GREEN, GREY = "#2c7fb8", "#c0392b", "#2e8b57", "#6b6b6b"


def save(fig, name, tight=True):
    kw = dict(bbox_inches="tight") if tight else {}
    fig.savefig(os.path.join(OUT, name + ".pdf"), **kw)
    fig.savefig(os.path.join(OUT, name + ".png"), **kw)
    plt.close(fig)
    print("wrote", name)


def clean3d(ax):
    """Reference-paper 3D furniture: pale panes, hairline grid, thin axes."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.962, 0.968, 0.976, 1.0))
        axis.pane.set_edgecolor((0.80, 0.80, 0.83, 1.0))
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update({"color": (0.78, 0.78, 0.82, 1.0),
                                     "linewidth": 0.35, "linestyle": "-"})
        axis._axinfo["axisline"].update({"color": (0.20, 0.20, 0.20, 1.0),
                                         "linewidth": 0.5})
    ax.tick_params(width=0.4, length=2.2, pad=1.5, labelsize=7.4)


# ----------------------------------------------------------------- fig 4
def fig4():
    b = pd.read_csv(os.path.join(OUT, "rev2b_books_arms.csv"))
    b["pen"] = (b.risk_mingap - b.risk_minrisk) * 100

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    for k, reg in enumerate(["adverse", "symmetric"]):
        a = ax[k]
        d = b[b.regime == reg]
        for arm, mk, fc, lab in [("base", "o", "white", "six-plan pool"),
                                 ("base+slack", "^", RED, "six plans + three slack")]:
            e = d[d.arm == arm]
            a.scatter(e.spread_teu, e.pen, s=26, marker=mk, facecolors=fc,
                      edgecolors=BLUE, linewidths=0.9, label=lab, zorder=3)
        x, y = d.spread_teu.values, d.pen.values
        A = np.vstack([np.ones_like(x), x]).T
        c0, c1 = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = A @ np.array([c0, c1])
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        xs = np.linspace(0, x.max() * 1.05, 40)
        a.plot(xs, c0 + c1 * xs, color=GREY, ls="--", lw=1.1, zorder=2)
        a.set_xlabel(r"commitment spread of the pool, $(v_{\max}-v_{\min})/\bar v$")
        a.set_ylabel("exposure penalty (pp)")
        a.set_title(f"({'ab'[k]})  {reg.capitalize()} drift", fontsize=9.5)
        a.text(0.04, 0.95,
               f"fit: {c0:.1f} + {c1:.0f}$\\times$spread\n$R^2$={r2:.2f}, n={len(d)} arms",
               transform=a.transAxes, fontsize=7.4, va="top", color=GREY)
        a.set_ylim(-1, max(b.pen.max() * 1.05, 4))
        if k == 0:
            a.legend(loc="lower right", fontsize=7.6)
    save(fig, "Figure_4")


# ----------------------------------------------------------------- fig 5
def fig5():
    r = pd.read_csv(os.path.join(OUT, "real_case_plans.csv"))
    kind = np.where(r.plan.str.startswith("llm"), "LLM",
                    np.where(r.plan.str.startswith("slack"), "deliberate slack", "heuristic"))
    col = {"heuristic": BLUE, "deliberate slack": GREEN, "LLM": RED}
    mk = {"heuristic": "o", "deliberate slack": "s", "LLM": "^"}
    lg = np.log10(r.gap.values)
    expo = r.risk.values
    vol = r.teu.values / 1000.0

    fig = plt.figure(figsize=(6.1, 4.6))
    ax = fig.add_axes([-0.05, 0.01, 0.94, 0.85], projection="3d")
    ax.computed_zorder = False

    # the diagonal: a least-squares plane through the nineteen points, drawn
    # only across the range the points themselves span; dense mesh on a pale
    # face, the surface idiom of the reference paper
    A = np.column_stack([lg, expo, np.ones_like(lg)])
    coef = np.linalg.lstsq(A, vol, rcond=None)[0]
    gx = np.linspace(lg.min(), lg.max(), 25)
    ey = np.linspace(expo.min(), expo.max(), 25)
    X, Y = np.meshgrid(gx, ey)
    ax.plot_surface(X, Y, coef[0] * X + coef[1] * Y + coef[2],
                    facecolor=(0.847, 0.890, 0.941, 1.0), alpha=1.0,
                    edgecolor=(0.208, 0.318, 0.494, 1.0), linewidth=0.22,
                    rstride=1, cstride=1, shade=False, zorder=1)

    for k in ["heuristic", "deliberate slack", "LLM"]:
        m = kind == k
        ax.scatter(lg[m], expo[m], vol[m], color=col[k], s=52, marker=mk[k],
                   edgecolors="k", linewidths=0.45, depthshade=False, label=k, zorder=5)

    i_gap = int(np.argmin(r.gap.values))
    i_risk = int(np.argmin(r.risk.values))
    ax.scatter([lg[i_gap]], [expo[i_gap]], [vol[i_gap]], s=150, facecolors="none",
               edgecolors=RED, linewidths=1.6, zorder=7, depthshade=False,
               label="gap-minimal plan")
    ax.scatter([lg[i_risk]], [expo[i_risk]], [vol[i_risk]], s=150, marker="s",
               facecolors="none", edgecolors=GREEN, linewidths=1.6, zorder=7,
               depthshade=False, label="exposure-minimal plan")

    ax.set_xlabel(r"$\log_{10}$ optimality gap  $g$", labelpad=7)
    ax.set_ylabel(r"realised exposure  $R$", labelpad=7)
    ax.set_zlabel("committed TEU (thousands)", labelpad=6)
    ax.set_zlim(0, 42)
    ax.set_zticks([0, 10, 20, 30, 40])
    ax.view_init(elev=21, azim=-64)
    ax.set_box_aspect((1.5, 1.0, 0.80), zoom=1.16)
    clean3d(ax)

    fig.suptitle("The nineteen plans of one voyage: the gap's pick and the safe\n"
                 "plan sit at opposite ends of realised exposure", x=0.44, y=0.99,
                 fontsize=9.2)
    ax.legend(loc="upper left", fontsize=7.2, bbox_to_anchor=(-0.03, 1.0),
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.35)
    save(fig, "Figure_5", tight=False)


# ----------------------------------------------------------------- fig 6
def fig6():
    """The lever mechanism, one instance per maritime family.

    The instance plotted is the one whose within-instance correlation between
    gap and exposure is closest to the family median (C: -0.775, D: -0.486);
    it is not the most favourable instance in the archive.
    """
    c = pd.read_csv(os.path.join(OUT, "llm_slot_candidates.csv"))
    c = c[c.feasible]
    d = pd.read_csv(os.path.join(OUT, "llm_bunker_candidates.csv"))

    def pick(df, xcol, rcol, icol, target):
        """Instance with rho closest to the family median, among pools of at
        least five plans in which both metrics take three or more values."""
        best, best_d = None, np.inf
        for i, g in df.groupby(icol):
            if len(g) < 5 or g[rcol].nunique() < 3 or g.gap.nunique() < 3:
                continue
            rho = spearmanr(g.gap.values, g[rcol].values).statistic
            if abs(rho - target) < best_d:
                best, best_d = i, abs(rho - target)
        return df[df[icol] == best].sort_values(xcol), best

    gc, ic = pick(c, "committed_teu", "overflow", "inst", -0.7746)
    gd, idd = pick(d, "lift_t", "overrun", "inst", -0.4857)

    panels = [(gc, "committed_teu", "committed TEU", "C  slots: rolled cargo",
               "overflow", "rolled cargo (overflow)", ic),
              (gd, "lift_t", "bunkers lifted (t)", "D  bunker: budget overrun",
               "overrun", "budget overrun", idd)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    for k, (g, xcol, xlab, ttl, rcol, rlab, inst) in enumerate(panels):
        a = ax[k]
        x = g[xcol].values
        a.plot(x, g.gap.values, "o-", color=BLUE, ms=5, lw=1.2,
               label="optimality gap", zorder=3)
        a.set_xlabel(xlab)
        a.set_ylabel("optimality gap", color=BLUE)
        a.tick_params(axis="y", colors=BLUE)
        a2 = a.twinx()
        a2.plot(x, g[rcol].values, "s--", color=RED, ms=5, lw=1.2,
                label="realised exposure", zorder=3)
        a2.set_ylabel(rlab, color=RED)
        a2.tick_params(axis="y", colors=RED)
        a2.spines["right"].set_visible(True)
        a2.spines["top"].set_visible(False)
        rho = spearmanr(g.gap.values, g[rcol].values).statistic
        a.set_title(f"({'ab'[k]})  {ttl}  (inst {int(inst)})", fontsize=9)
        a.set_zorder(a2.get_zorder() + 1)
        a.patch.set_visible(False)
        if k == 0:
            h1, l1 = a.get_legend_handles_labels()
            h2, l2 = a2.get_legend_handles_labels()
            handles, labels = h1 + h2, l1 + l2
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7.4,
               bbox_to_anchor=(0.5, -0.01))
    save(fig, "Figure_6")


# ----------------------------------------------------------------- fig 7
def fig7():
    e = pd.read_csv(os.path.join(OUT, "effect_ci.csv"))
    e = e.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.5), layout="constrained")
    for i, row in e.iterrows():
        ax.plot([row.lo * 100, row.hi * 100], [i, i], color="0.62", lw=5.2,
                solid_capstyle="butt", zorder=1)
        ax.plot([row.lo_b * 100, row.hi_b * 100], [i, i], color=BLUE, lw=1.8,
                solid_capstyle="butt", zorder=2)
        ax.plot([row.penalty * 100], [i], "o", ms=5.2, color=RED, mec="k", mew=0.4, zorder=3)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(range(len(e)))
    ax.set_yticklabels([f"{r.row}  (n={int(r.n)})" for _, r in e.iterrows()], fontsize=8)
    ax.set_xlabel("exposure penalty, gap-minimal minus safest plan (pp)")
    ax.set_ylim(-0.7, len(e) - 0.3)
    ax.plot([], [], color="0.62", lw=5.2, label="Wilson 95 % interval")
    ax.plot([], [], color=BLUE, lw=1.8, label="instance bootstrap 95 % interval")
    ax.plot([], [], "o", color=RED, mec="k", label="point estimate")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.33), fontsize=7.6)
    ax.set_title("All nine contrasts keep the sign of the effect")
    save(fig, "Figure_7")


# ----------------------------------------------------------------- fig 8
def fig8():
    d = pd.read_csv(os.path.join(OUT, "rev2_drift.csv"))
    d["pen"] = (d.risk_mingap - d.risk_minrisk) * 100
    g = d.groupby(["fam", "regime"]).agg(pen=("pen", "median"),
                                         conf=("conflict", "mean"),
                                         n=("conflict", "count")).reset_index()
    fams, regs = ["C", "D"], ["adverse", "symmetric", "favourable"]

    fig = plt.figure(figsize=(7.8, 3.9))
    ax = fig.add_axes([0.015, 0.02, 0.52, 0.86], projection="3d")
    ax.computed_zorder = False
    xs, ys, zs, ns = [], [], [], []
    for i, f in enumerate(fams):
        for j, r in enumerate(regs):
            row = g[(g.fam == f) & (g.regime == r)]
            xs.append(i)
            ys.append(j)
            zs.append(float(row.pen.iloc[0]) if len(row) else np.nan)
            ns.append(int(row.n.iloc[0]) if len(row) else 0)
    xs, ys = np.array(xs), np.array(ys)
    top = np.nanmax(zs)
    bar_cmap = plt.get_cmap("Blues")
    for x, y, z, n in zip(xs, ys, zs, ns):
        if n and np.isfinite(z):
            ax.plot([x, x], [y, y], [0, z], color=bar_cmap(0.30 + 0.60 * z / top),
                    lw=3.4, solid_capstyle="round", zorder=4)
            ax.scatter([x], [y], [z], s=78, color=bar_cmap(0.50 + 0.50 * z / top),
                       edgecolors="k", linewidths=0.5, depthshade=False, zorder=6)
            ax.text(x, y - 0.13, z + top * 0.07, f"{z:.1f}", ha="center",
                    va="bottom", fontsize=7.4, zorder=8,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
        else:
            ax.plot([x, x], [y, y], [0, top * 0.06], color="0.6", lw=1.0, ls=":",
                    zorder=4)
            ax.scatter([x], [y], [top * 0.06], s=42, facecolors="white",
                       edgecolors="0.55", linewidths=0.7, depthshade=False, zorder=6)
    ax.set_zlim(0, top * 1.30)
    ax.set_zticks([0, 1, 2])
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(["C slots", "D bunker"], fontsize=8.0)
    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels(["adverse", "symmetric", "favourable"], fontsize=7.6)
    ax.set_zlabel("median penalty (pp)", labelpad=1, fontsize=8.2)
    ax.view_init(elev=20, azim=-58)
    ax.set_box_aspect((1.25, 0.95, 0.85), zoom=1.18)
    clean3d(ax)
    ax.set_title("(a)  Penalty by drift regime", fontsize=9.5, pad=2)

    a = fig.add_axes([0.70, 0.17, 0.29, 0.75])
    w = 0.36
    for i, f in enumerate(fams):
        for j, r in enumerate(regs):
            row = g[(g.fam == f) & (g.regime == r)]
            v = float(row.conf.iloc[0]) * 100 if len(row) else np.nan
            if not np.isfinite(v):
                a.text(j + (i - 0.5) * w, 4, "n/a", ha="center", va="bottom",
                       fontsize=7.2, color="0.45")
                continue
            a.bar(j + (i - 0.5) * w, v, w, color=BLUE if f == "C" else RED,
                  alpha=0.92, edgecolor="white", linewidth=0.4)
            a.text(j + (i - 0.5) * w, v + 1.8, f"{v:.1f}", ha="center", fontsize=7.2)
    a.set_xticks(range(len(regs)))
    a.set_xticklabels(["adverse", "symmetric", "favourable"], fontsize=8)
    a.set_ylabel("conflict rate (%)")
    a.set_ylim(0, 100)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_title("(b)  Conflict rate by regime", fontsize=9.5)
    save(fig, "Figure_8")


# ----------------------------------------------------------------- fig 9
def fig9():
    s = pd.read_csv(os.path.join(OUT, "screen_ci.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 4.7), layout="constrained")
    for _, r in s.iterrows():
        col = BLUE if r.family == "C" else RED
        xq = r.n_rel_2d / r.n_plans * 100
        xg = r.n_rel_gap / r.n_plans * 100
        ax.annotate("", xy=(xq, r.fr_2d * 100), xytext=(xg, r.fr_gap * 100),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, alpha=0.75,
                                    shrinkA=5, shrinkB=5))
        ax.errorbar(xg, r.fr_gap * 100,
                    yerr=[[r.fr_gap * 100 - r.lo_gap * 100], [r.hi_gap * 100 - r.fr_gap * 100]],
                    fmt="o", ms=6.5, color=col, mec="k", mew=0.4, capsize=2.2, lw=1.1, zorder=3)
        if np.isnan(r.lo_2d):
            ax.plot(xq, r.fr_2d * 100, "s", ms=6.5, mfc="white", mec=col, mew=1.4, zorder=3)
            ax.annotate("1 of 107 released", xy=(xq, r.fr_2d * 100), xytext=(14, -10),
                        textcoords="offset points", fontsize=7.4, color=col,
                        va="center",
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
        else:
            ax.errorbar(xq, r.fr_2d * 100,
                        yerr=[[r.fr_2d * 100 - r.lo_2d * 100], [r.hi_2d * 100 - r.fr_2d * 100]],
                        fmt="s", ms=6.5, color=col, mfc="white", mec=col, mew=1.4,
                        capsize=2.2, lw=1.1, zorder=3)
        dy = 6 if r.family == "C" else -13
        ax.annotate(rf"$\tau$={r.tau:g}", xy=(xg, r.fr_gap * 100), xytext=(-11, dy),
                    textcoords="offset points", fontsize=7, color=col, ha="right")
    ax.axhline(25, color=GREY, ls="--", lw=0.9, zorder=0)
    ax.text(61, 27, "random release (25 %)", fontsize=7, color=GREY, ha="right")
    ax.set_xlabel("share of the pool released (%)")
    ax.set_ylabel("false-release rate (%)")
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 108)
    ax.set_title("The exposure quantile buys safety with volume")
    ax.plot([], [], "o", color=GREY, mec="k", label="gap threshold only")
    ax.plot([], [], "s", color=GREY, mfc="white", mew=1.3, label="gap and exposure quantile")
    ax.plot([], [], color=GREY, lw=1.0, label="adding the quantile")
    ax.legend(loc="upper right", fontsize=7.2)
    save(fig, "Figure_9")


# ----------------------------------------------------------------- fig 10
def fig10():
    t = pd.read_csv(os.path.join(OUT, "tie_pool.csv")).dropna(subset=["gap_spread"])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    a = ax[0]
    a.scatter(t.gap_spread, t.expo_spread, s=34, color=RED, edgecolors="k", linewidths=0.4)
    xs = np.linspace(0.06, 0.11, 30)
    ratio = float(np.median(t.expo_spread) / np.median(t.gap_spread))
    a.plot(xs, ratio * xs, color=GREY, ls="--", lw=1.0)
    a.text(0.03, 0.96, f"median ratio {ratio:.0f}$\\times$", transform=a.transAxes,
           ha="left", va="top", fontsize=7.8, color=GREY)
    a.set_xlabel("gap spread within the tied pool (pp)")
    a.set_ylabel("exposure spread within the pool (pp)")
    a.set_title(f"(a)  {len(t)} pools the solver cannot separate")
    a.set_xlim(0.06, 0.11)

    a = ax[1]
    a.scatter(t.teu_spread, t.expo_spread, s=34, color=BLUE, edgecolors="k", linewidths=0.4)
    rho = spearmanr(t.teu_spread, t.expo_spread).statistic
    a.text(0.04, 0.94, f"$\\rho$={rho:+.2f}", transform=a.transAxes, fontsize=7.6,
           color="0.25", va="top")
    a.set_xlabel("committed TEU spread within the pool")
    a.set_ylabel("exposure spread (pp)")
    a.set_title("(b)  Volume is what separates them")
    save(fig, "Figure_10")


if __name__ == "__main__":
    for f in (fig4, fig5, fig6, fig7, fig8, fig9, fig10):
        f()
    print("done")
