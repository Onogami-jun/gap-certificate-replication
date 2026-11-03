# -*- coding: utf-8 -*-
"""
Rev-2 experiments responding to the referee report.

E1  drift symmetry: adverse / mean-zero / favourable, for C (slots) and D (bunker).
E2  baselines: nominal optimum, Bertsimas-Sim budgeted robust (C), two-stage SAA (C),
    lift-hedging at the requirement quantile (D).  Evaluated on the SAME adverse scenarios.
E3  clean twenty-book ensemble with no deliberate-slack plans.
E4  dollar denominators: committed revenue (C) and committed cost (D).

Outputs: rev2_drift.csv, rev2_baselines.csv, rev2_books_noslack.csv,
         rev2_dollars.csv, rev2_summary.json
"""
import json
import os
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import spearmanr

import real_case as rc

OUT = os.path.dirname(os.path.abspath(__file__))


def load_functions(path, stop_marker):
    """exec the source up to the analysis block so we get the generators, not their run."""
    src = open(path, encoding="utf-8").read()
    cut = src.index(stop_marker)
    g = {"__name__": "mod", "__file__": path}
    exec(compile(src[:cut], path, "exec"), g)
    return g


SL = load_functions(os.path.join(OUT, "sea_slot_pilot.py"), "inst_rows, cand_rows = []")
BK = load_functions(os.path.join(OUT, "sea_bunker_pilot.py"), "inst_rows, cand_rows = []")
M, BUNK_RATE, SPOT = BK["M"], BK["BUNK_RATE"], BK["SPOT_PREMIUM"]


def u_eff(rng, n, regime):
    u = rng.uniform(0.0, 1.0, n)
    if regime == "adverse":
        return u
    if regime == "symmetric":
        return 2.0 * u - 1.0
    return -u                                   # favourable


# ---------------------------------------------------------------- E1 + risk
def slot_risk(inst, x, regime="adverse", dls=(0.10, 0.20), K=10):
    n, L = inst["n"], SL["N_LEGS"]
    com = float((inst["q"] * x).sum())
    ov, bad = [], []
    for di, dl in enumerate(dls):
        for k in range(K):
            rng = np.random.default_rng(zlib.crc32(f"{regime}|{di}|{inst['o'].tolist()}|{k}".encode()))
            mq = 1 + dl * u_eff(rng, n, regime)
            mc = 1 - dl * u_eff(rng, L, regime)
            load = (inst["w"] * (inst["q"] * mq * x)[:, None]).sum(0)
            over = np.maximum(0.0, load - inst["caps"] * mc)
            ov.append(float(over.max() / max(1e-9, com)))
            bad.append(float(np.any(over > 1e-9)))
    return float(np.mean(ov)), float(np.mean(bad))


def bunker_risk(inst, z, y, regime="adverse", dls=(0.10, 0.20), K=10):
    base = BK["eval_cost"](inst, z, y)
    out, absv = [], []
    for di, dl in enumerate(dls):
        for k in range(K):
            rng = np.random.default_rng(zlib.crc32(f"{regime}|{di}|{inst['p'].round(4)}|{k}".encode()))
            p2 = inst["p"] * (1 + dl * u_eff(rng, M, regime))
            cap2 = inst["cap"] * (1 - dl * u_eff(rng, M, regime))
            B2 = inst["B"] * (1 + dl * float(u_eff(rng, 1, regime)[0]))
            got = float(np.minimum(y, cap2 * z).sum())
            short = max(0.0, B2 - got)
            c2 = float((p2 * np.minimum(y, cap2 * z)).sum()
                       + sum(BK["fixed_cost"](inst, j) for j in range(M) if z[j] == 1)
                       + short * SPOT * p2.max())
            out.append((c2 - base) / base)
            absv.append(c2 - base)
    return float(np.mean(out)), float(np.mean(absv))


# ---------------------------------------------------------------- E2 baselines (C)
def slot_saa(inst, S=12, dl=0.20, lam_bump=0.30, seed=0):
    """Two-stage SAA: max E[revenue] - 0.3 * rate * E[rolled TEU]."""
    n, L = inst["n"], SL["N_LEGS"]
    A = inst["A"]                                    # (legs, items), normalised load
    rng = np.random.default_rng(zlib.crc32(f"saa|{inst['o'].tolist()}".encode()))
    mq = 1 + dl * u_eff(rng, S, "adverse")           # one volume multiplier per scenario
    mc = 1 - dl * u_eff(rng, S * L, "adverse").reshape(S, L)
    rate0 = float(np.median(inst["r"]))
    nx, no = n, L * S
    c = np.concatenate([-inst["r"] * inst["q"], np.full(no, lam_bump * rate0 / S)])
    integ = np.concatenate([np.ones(nx), np.zeros(no)])
    rows, lo, hi = [], [], []
    # nominal capacity
    for l in range(L):
        a = np.zeros(nx + no); a[:n] = A[l]
        rows.append(a); lo.append(-np.inf); hi.append(1.0)
    # o_{l,s} - m_s * A[l] x >= -cap_{l,s}
    for s in range(S):
        for l in range(L):
            a = np.zeros(nx + no)
            a[:n] = -mq[s] * A[l]
            a[nx + l * S + s] = 1.0
            rows.append(a); lo.append(-mc[s, l]); hi.append(np.inf)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=integ, bounds=Bounds(np.zeros(nx + no), np.concatenate(
                   [np.ones(nx), np.full(no, np.inf)])))
    if not res.success:
        return None
    return np.round(res.x[:n]).astype(int)


def slot_robust(inst, gamma=4.4, d=0.20):
    """Bertsimas-Sim budgeted uncertainty on the leg loads."""
    n, L = inst["n"], SL["N_LEGS"]
    A = inst["A"]
    nx, nz, npv = n, L, n * L
    c = np.concatenate([-inst["r"] * inst["q"], np.zeros(nz + npv)])
    integ = np.concatenate([np.ones(nx), np.zeros(nz + npv)])
    rows, lo, hi = [], [], []
    for l in range(L):
        a = np.zeros(nx + nz + npv)
        a[:n] = A[l]
        a[nx + l] = gamma
        a[nx + nz + l * n:(nx + nz + (l + 1) * n)] = 1.0
        rows.append(a); lo.append(-np.inf); hi.append(1.0)
    for l in range(L):
        for i in range(n):
            a = np.zeros(nx + nz + npv)
            a[nx + l] = 1.0
            a[nx + nz + l * n + i] = 1.0
            a[i] = -d * A[l, i]
            rows.append(a); lo.append(0.0); hi.append(np.inf)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=integ,
               bounds=Bounds(np.zeros(nx + nz + npv),
                             np.concatenate([np.ones(nx), np.full(nz + npv, np.inf)])))
    if not res.success:
        return None
    return np.round(res.x[:n]).astype(int)


# ---------------------------------------------------------------- E1 drift table
drift_rows = []
for s in range(40):
    inst = SL["gen_slot"](1000 + s, n_book=int(18 + 6 * (s % 3)), cap=8000,
                         tight=[1.05, 1.15, 1.30][s % 3])
    opt, _ = SL["solve"](inst)
    if not np.isfinite(opt):
        continue
    inst["opt"] = opt
    cs = SL["candidates"](inst)
    if len(cs) < 2:
        continue
    for regime in ("adverse", "symmetric", "favourable"):
        g, r = [], []
        for val, nm, x in cs:
            g.append((opt - val) / abs(opt))
            r.append(slot_risk(inst, x, regime)[0])
        g, r = np.array(g), np.array(r)
        if np.ptp(r) <= 1e-12:
            drift_rows.append({"fam": "C", "inst": 1000 + s, "regime": regime, "n": len(cs),
                               "rho": np.nan, "conflict": np.nan, "risk_mingap": r.min(),
                               "risk_minrisk": r.min(), "note": "no risk variation"})
            continue
        ig, ir = int(np.argmin(g)), int(np.argmin(r))
        drift_rows.append({"fam": "C", "inst": 1000 + s, "regime": regime, "n": len(cs),
                           "rho": spearmanr(g, r).statistic, "conflict": float(ig != ir),
                           "risk_mingap": r[ig], "risk_minrisk": r[ir], "note": ""})

for s in range(40):
    inst = BK["gen"](2000 + s)
    opt, zo, yo = BK["solve"](inst)
    if not np.isfinite(opt):
        continue
    cs = BK["candidates"](inst)
    if len(cs) < 2:
        continue
    for regime in ("adverse", "symmetric", "favourable"):
        g, r = [], []
        for cost, nm, z, y in cs:
            g.append((cost - opt) / abs(opt))
            r.append(bunker_risk(inst, z, y, regime)[0])
        g, r = np.array(g), np.array(r)
        if np.ptp(r) <= 1e-12:
            drift_rows.append({"fam": "D", "inst": 2000 + s, "regime": regime, "n": len(cs),
                               "rho": np.nan, "conflict": np.nan, "risk_mingap": r.min(),
                               "risk_minrisk": r.min(), "note": "no risk variation"})
            continue
        ig, ir = int(np.argmin(g)), int(np.argmin(r))
        drift_rows.append({"fam": "D", "inst": 2000 + s, "regime": regime, "n": len(cs),
                           "rho": spearmanr(g, r).statistic, "conflict": float(ig != ir),
                           "risk_mingap": r[ig], "risk_minrisk": r[ir], "note": ""})

DR = pd.DataFrame(drift_rows)
DR.to_csv(os.path.join(OUT, "rev2_drift.csv"), index=False)
print("== E1 drift symmetry ==")
for fam in ("C", "D"):
    for reg in ("adverse", "symmetric", "favourable"):
        d = DR[(DR.fam == fam) & (DR.regime == reg)]
        ok = d[d.conflict.notna()]
        print(f"  {fam} {reg:11s}: n_all={len(d):2d} usable={len(ok):2d} "
              f"conflict={ok.conflict.mean() if len(ok) else float('nan'):.3f} "
              f"rho_med={ok.rho.median() if len(ok) else float('nan'):+.3f} "
              f"mingap_risk={ok.risk_mingap.median() if len(ok) else float('nan'):.4f} "
              f"minrisk_risk={ok.risk_minrisk.median() if len(ok) else float('nan'):.4f}")

# ---------------------------------------------------------------- E2 baselines
base_rows = []
dollar_rows = []
for s in range(40):
    inst = SL["gen_slot"](1000 + s, n_book=int(18 + 6 * (s % 3)), cap=8000,
                         tight=[1.05, 1.15, 1.30][s % 3])
    opt, xopt = SL["solve"](inst)
    if not np.isfinite(opt):
        continue
    inst["opt"] = opt
    cs = SL["candidates"](inst)
    if len(cs) < 2:
        continue
    plans = {"nominal-opt": xopt, "robust-G2.2": slot_robust(inst, 2.2),
             "robust-G4.4": slot_robust(inst, 4.4), "SAA": slot_saa(inst)}
    vals = {nm: float(inst["r"] @ np.asarray(x, float)) for nm, x in plans.items() if x is not None}
    vals["best-candidate"] = max(v for v, _, _ in cs)
    for nm, x in plans.items():
        if x is None:
            continue
        obj = float(inst["r"] @ x)
        rk, _ = slot_risk(inst, x, "adverse")
        base_rows.append({"fam": "C", "inst": 1000 + s, "policy": nm,
                          "gap": (opt - obj) / abs(opt), "risk": rk, "obj": obj})
    # C's objective is in normalised revenue units, not USD: record the denominator only.
    for nm in ("nominal-opt", "robust-G4.4"):
        x = plans.get(nm)
        if x is None:
            continue
        obj = float(inst["r"] @ x)
        rk, _ = slot_risk(inst, x, "adverse")
        dollar_rows.append({"fam": "C", "policy": nm, "committed": obj, "risk": rk,
                            "exposure_abs": rk * obj})

for s in range(40):
    inst = BK["gen"](2000 + s)
    opt, zo, yo = BK["solve"](inst)
    if not np.isfinite(opt):
        continue
    base_rows.append({"fam": "D", "inst": 2000 + s, "policy": "nominal-opt",
                      "gap": 0.0, "risk": bunker_risk(inst, zo, yo)[0],
                      "obj": BK["eval_cost"](inst, zo, yo)})
    for q in (0.5, 0.75, 0.9, 0.95):
        scale = 1 + 0.20 * q
        y2 = np.minimum(inst["cap"], yo * scale)
        if y2.sum() < inst["B"]:
            continue
        z2 = (y2 > 1e-9).astype(int)
        cost2 = BK["eval_cost"](inst, z2, y2)
        base_rows.append({"fam": "D", "inst": 2000 + s, "policy": f"hedge-q{int(q*100)}",
                          "gap": (cost2 - opt) / abs(opt),
                          "risk": bunker_risk(inst, z2, y2)[0], "obj": cost2})
    _, ab0 = bunker_risk(inst, zo, yo)
    dollar_rows.append({"fam": "D", "policy": "nominal-opt", "committed": opt,
                        "risk": bunker_risk(inst, zo, yo)[0], "exposure_abs": ab0})
    for q in (0.75, 0.9):
        scale = 1 + 0.20 * q
        y2 = np.minimum(inst["cap"], yo * scale)
        if y2.sum() < inst["B"]:
            continue
        z2 = (y2 > 1e-9).astype(int)
        cost2 = BK["eval_cost"](inst, z2, y2)
        rk, ab = bunker_risk(inst, z2, y2)
        dollar_rows.append({"fam": "D", "policy": f"hedge-q{int(q*100)}", "committed": cost2,
                            "risk": rk, "exposure_abs": ab})

BR = pd.DataFrame(base_rows)
BR.to_csv(os.path.join(OUT, "rev2_baselines.csv"), index=False)
print("\n== E2 baselines ==")
for fam in ("C", "D"):
    for pol, d in BR[BR.fam == fam].groupby("policy"):
        print(f"  {fam} {pol:16s} n={len(d):2d} gap_med={d.gap.median():.4f} "
              f"risk_med={d.risk.median():.4f} risk_mean={d.risk.mean():.4f}")

DL = pd.DataFrame(dollar_rows)
DL.to_csv(os.path.join(OUT, "rev2_dollars.csv"), index=False)
print("\n== E4 denominators ==")
for fam in ("C", "D"):
    for pol, d in DL[DL.fam == fam].groupby("policy"):
        unit = "USD" if fam == "D" else "obj-units"
        print(f"  {fam} {pol:14s} committed_med={d.committed.median():,.0f} "
              f"risk_med={d.risk.median():.4f} "
              f"exposure_med={d.exposure_abs.median():,.0f} [{unit}]")

# ---------------------------------------------------------------- E3 clean books
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


book_rows = []
for b in range(1, 21):
    book = rc.gen_book(b)
    opt, xopt, ysg0, yrt0 = rc.joint_opt(book, np.linspace(rc.B_SG_LEG, rc.TANK["Singapore"], 9),
                                         np.linspace(0.0, rc.TANK["Rotterdam"], 7))
    if not np.isfinite(opt):
        continue
    rev_opt = float(book["r"] * book["q"] @ xopt)
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
            if float((book["w"] * (book["q"] * x)[:, None]).sum(0).max()) > rc.CAP_TEU.max() + 1e-6:
                continue
            obj = float(book["r"] * book["q"] @ x) - (rc.VLSFO["Singapore"] * ysg
                                                      + rc.VLSFO["Rotterdam"] * yrt)
            plans.append((obj, f"{kn}|{bn}", x, ysg, yrt))
    if len(plans) < 3:
        continue
    for regime in ("adverse", "symmetric", "favourable"):
        g = np.array([(opt - o) / abs(opt) for o, _, _, _, _ in plans])
        r = np.array([book_risk(book, x, a, b_, regime) for _, _, x, a, b_ in plans])
        if np.ptp(r) <= 1e-12:
            continue
        ig, ir = int(np.argmin(g)), int(np.argmin(r))
        book_rows.append({"book": b, "regime": regime, "n": len(plans),
                          "rho": spearmanr(g, r).statistic,
                          "conflict": float(ig != ir), "risk_mingap": r[ig],
                          "risk_minrisk": r[ir], "gap_mingap": g[ig], "gap_minrisk": g[ir]})
NB = pd.DataFrame(book_rows)
NB.to_csv(os.path.join(OUT, "rev2_books_noslack.csv"), index=False)
print("\n== E3 clean 20-book ensemble (no slack plans) ==")
for reg, d in NB.groupby("regime"):
    print(f"  {reg:11s} books={d.book.nunique():2d} conflict={d.conflict.mean():.3f} "
          f"rho_med={d.rho.median():+.3f} mingap={d.risk_mingap.median():.4f} "
          f"minrisk={d.risk_minrisk.median():.4f}")

summary = {
    "drift": {f"{fam}|{reg}": {
        "n": int(len(d)), "usable": int(d.conflict.notna().sum()),
        "conflict": None if d.conflict.notna().sum() == 0 else float(d.conflict.mean()),
        "rho_med": None if d.rho.notna().sum() == 0 else float(d.rho.median()),
        "risk_mingap": None if d.risk_mingap.notna().sum() == 0 else float(d.risk_mingap.median()),
        "risk_minrisk": None if d.risk_minrisk.notna().sum() == 0 else float(d.risk_minrisk.median()),
    } for (fam, reg), d in DR.groupby(["fam", "regime"])},
    "baselines": {f"{fam}|{pol}": {
        "n": int(len(d)), "gap_med": float(d.gap.median()), "risk_med": float(d.risk.median()),
        "risk_mean": float(d.risk.mean())} for (fam, pol), d in BR.groupby(["fam", "policy"])},
    "books": {reg: {"books": int(d.book.nunique()), "conflict": float(d.conflict.mean()),
                    "rho_med": float(d.rho.median()),
                    "risk_mingap": float(d.risk_mingap.median()),
                    "risk_minrisk": float(d.risk_minrisk.median()),
                    "gap_mingap": float(d.gap_mingap.median()),
                    "gap_minrisk": float(d.gap_minrisk.median())}
              for reg, d in NB.groupby("regime")},
}
with open(os.path.join(OUT, "rev2_summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
print("\nsaved rev2_summary.json")
