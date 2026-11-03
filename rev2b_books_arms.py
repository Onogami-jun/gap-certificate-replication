# -*- coding: utf-8 -*-
"""
Controlled lever-spread experiment on the twenty booking books.

Arm 1  base     : the six heuristic plans of rev2 (two slot-ordering rules x
                  three bunker strategies), every plan loaded to capacity.
Arm 2  base+slack: the same six plans plus three deliberate-slack plans
                  (95 / 90 / 80 % of capacity), which span the commitment level
                  by construction.

Everything else is held fixed (same books, same observation, same drift draws),
so the difference between arms is the spread of committed volume in the pool.
Outputs rev2b_books_arms.csv.
"""
import os
import zlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import real_case as rc

OUT = os.path.dirname(os.path.abspath(__file__))


def u_eff(rng, n, regime):
    u = rng.uniform(0.0, 1.0, n)
    if regime == "adverse":
        return u
    if regime == "symmetric":
        return 2.0 * u - 1.0
    return -u


def book_risk(book, x, ysg, yrt, regime="adverse", dls=(0.10, 0.20), ns=100):
    rev = float(book["r"] * book["q"] @ x)
    fees = float(rc.PORT_FEE.sum())
    committed_cost = rc.VLSFO["Singapore"] * ysg + rc.VLSFO["Rotterdam"] * yrt + fees
    margin = rev - committed_cost
    out = []
    for di, dl in enumerate(dls):
        for s in range(ns):
            rng = np.random.default_rng(zlib.crc32(f"{regime}|{di}|{s}|{ysg:.1f}|{yrt:.1f}".encode()))
            mq = 1 + dl * u_eff(rng, book["n"], regime)
            mc = 1 - dl * u_eff(rng, 4, regime)
            mp_sg = 1 + dl * float(u_eff(rng, 1, regime)[0])
            mp_rt = 1 + dl * float(u_eff(rng, 1, regime)[0])
            mb = 1 + dl * float(u_eff(rng, 1, regime)[0])
            q2 = book["q"] * mq
            caps2 = rc.CAP_TEU * mc
            caps2[3] = max(1000.0, caps2[3] - rc.fuel_draft_penalty(ysg, yrt))
            p2 = {"Singapore": rc.VLSFO["Singapore"] * mp_sg,
                  "Rotterdam": rc.VLSFO["Rotterdam"] * mp_rt}
            b2 = rc.B_TOTAL * mb
            load = (book["w"] * (q2 * x)[:, None]).sum(0)
            over = np.maximum(0.0, load - caps2)
            bump = float(min(1.0, over.max() / max(1e-9, load.max())))
            lost = bump * rev
            got = min(ysg, rc.TANK["Singapore"]) + min(yrt, rc.TANK["Rotterdam"])
            short = max(0.0, b2 - got)
            fuel = (p2["Singapore"] * min(ysg, rc.TANK["Singapore"])
                    + p2["Rotterdam"] * min(yrt, rc.TANK["Rotterdam"])
                    + short * rc.SPOT_PREMIUM * max(p2.values()))
            realized = (rev - lost) - fuel - fees - rc.BUMP_PENALTY * lost
            out.append((margin - realized) / abs(margin))
    return float(np.mean(out))


def base_plans(book):
    rev_opt = None
    bunkers = {"minlift": (rc.B_SG_LEG, rc.TANK["Rotterdam"]),
               "balanced": (0.5 * (rc.B_SG_LEG + rc.TANK["Singapore"]), 400.0),
               "tankfull": (rc.TANK["Singapore"], 0.0)}
    keys = {"rate": book["r"],
            "rateshort": book["r"] / np.maximum(1.0, book["d"] - book["o"])}
    plans = []
    for kn, key in keys.items():
        for bn, (ysg, yrt) in bunkers.items():
            pen = rc.fuel_draft_penalty(ysg, yrt)
            x = rc.greedy_slots(book, key, penalty=pen)
            load = (book["w"] * (book["q"] * x)[:, None]).sum(0)
            if load.max() > rc.CAP_TEU.max() + 1e-6:
                continue
            obj = float(book["r"] * book["q"] @ x) - (rc.VLSFO["Singapore"] * ysg
                                                      + rc.VLSFO["Rotterdam"] * yrt)
            plans.append((obj, kn, bn, x, ysg, yrt))
    return plans


def slack_plans(book):
    out = []
    ysg = 0.5 * (rc.B_SG_LEG + rc.TANK["Singapore"])
    yrt = 400.0
    pen = rc.fuel_draft_penalty(ysg, yrt)
    for frac in (0.95, 0.90, 0.80):
        x = rc.greedy_slots(book, book["r"], penalty=pen, fill_frac=frac)
        if not x.any():
            continue
        obj = float(book["r"] * book["q"] @ x) - (rc.VLSFO["Singapore"] * ysg
                                                  + rc.VLSFO["Rotterdam"] * yrt)
        out.append((obj, f"slack{int(frac*100)}", "balanced", x, ysg, yrt))
    return out


rows = []
for b in range(1, 21):
    book = rc.gen_book(b)
    opt, xopt, ysg0, yrt0 = rc.joint_opt(book, np.linspace(rc.B_SG_LEG, rc.TANK["Singapore"], 9),
                                         np.linspace(0.0, rc.TANK["Rotterdam"], 7))
    if not np.isfinite(opt):
        continue
    base = base_plans(book)
    slack = slack_plans(book)
    if len(base) < 3 or len(slack) < 2:
        continue
    for regime in ("adverse", "symmetric"):
        for arm, plans in (("base", base), ("base+slack", base + slack)):
            g = np.array([(opt - o) / abs(opt) for o, _, _, _, _, _ in plans])
            r = np.array([book_risk(book, x, a, c, regime) for _, _, _, x, a, c in plans])
            if np.ptp(r) <= 1e-12:
                continue
            ig, ir = int(np.argmin(g)), int(np.argmin(r))
            v = np.array([float(book["q"] @ x) for _, _, _, x, _, _ in plans])
            spread = (v.max() - v.min()) / max(1e-9, v.mean())
            rows.append({"book": b, "regime": regime, "arm": arm, "n": len(plans),
                         "spread_teu": spread, "rho": spearmanr(g, r).statistic,
                         "conflict": float(ig != ir), "risk_mingap": r[ig],
                         "risk_minrisk": r[ir], "gap_mingap": g[ig], "gap_minrisk": g[ir],
                         "v_teu_mingap": v[ig], "v_teu_minrisk": v[ir]})

A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT, "rev2b_books_arms.csv"), index=False)
print("== lever-spread arms ==")
for reg, d in A.groupby("regime"):
    for arm, e in d.groupby("arm"):
        print(f"  {reg:9s} {arm:11s} books={e.book.nunique():2d} n_plans={e.n.median():.0f} "
              f"spread={e.spread_teu.median():.3f} conflict={e.conflict.mean():.3f} "
              f"rho={e.rho.median():+.3f} mingap={e.risk_mingap.median():.4f} "
              f"minrisk={e.risk_minrisk.median():.4f} "
              f"excess={(e.risk_mingap - e.risk_minrisk).median():+.4f}")
