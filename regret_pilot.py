#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pilot experiment: does the duality-gap certificate rank *decisions* correctly?

For every instance in results_v2/candidates_labeled.csv we

  1. rebuild the exact instance from run_real.GENERATORS (same seeds),
  2. perturb the decision-relevant parameters (delta in {5%,10%,20%}, K scenarios),
  3. score every nominal-feasible candidate x by its *realized* regret
        regret(x) = (f_realized(R(x)) - f_realized(x*_realized)) / |f_realized(x*_realized)|
     where R is a one-sided (monotone-safe) repair policy,
  4. compare the gap-ordering of candidates with the regret-ordering.

Outputs (written next to this script):
  regret_pilot_candidates.csv   one row per (instance, candidate)
  regret_pilot_instances.csv    one row per instance
  regret_pilot_summary.json     aggregate per problem type
"""
import argparse, copy, json, os, sys, time, zlib
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import spearmanr

PAPER = r"C:\Users\D0gSXG\Desktop\project\论文\paper_dual_gap"
EXP = os.path.join(PAPER, "experiments")
sys.path.insert(0, EXP)
import run_real as R  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(EXP, "results_v2", "candidates_labeled.csv")

BASE_SEED = {"Knapsack": 1000, "Portfolio": 2000, "Diet": 3000, "SetCover": 4000,
             "BinPacking": 5000, "TSP": 6000, "Scheduling": 7000, "Facility": 8000}

INF = np.inf


# ---------------------------------------------------------------- rebuild
def get_problem(ptype, inst):
    seed = int(str(inst).split("_s")[-1])
    i = seed - BASE_SEED[ptype]
    p = R.GENERATORS[ptype](i)
    if p.name != inst:
        raise ValueError(f"name mismatch: {p.name} != {inst}")
    return p


def solve_opt(q):
    """Exact optimum of q in its own sense (no LP bound needed)."""
    sign = 1.0 if q.sense == "min" else -1.0
    cons = []
    if len(q.b_ub):
        cons.append(LinearConstraint(q.A_ub, -INF, q.b_ub))
    if len(q.b_eq):
        cons.append(LinearConstraint(q.A_eq, q.b_eq, q.b_eq))
    r = milp(sign * q.c, constraints=cons, integrality=q.integrality,
             bounds=Bounds(q.lb, q.ub), options={"time_limit": 30, "mip_rel_gap": 0})
    return float(sign * r.fun) if r.success else np.nan


# ------------------------------------------------------------- scenario
def make_scenario(p, ptype, delta, rng, mode="symmetric"):
    """Return a copy of p with perturbed decision-relevant parameters.

    mode="symmetric": xi ~ U(-1,1) (zero-mean drift)
    mode="adverse":   xi ~ U(0,1)  in the direction that hurts a tight decision
                      (weights/risks/times/costs up, nutrient content down)
    """
    q = copy.deepcopy(p)
    shape = p.A_ub.shape
    if mode == "adverse":
        def noise(s): return rng.uniform(0, 1, s)
        sgn = 1.0
    else:
        def noise(s): return rng.uniform(-1, 1, s)
        sgn = 1.0

    if ptype == "Knapsack":                      # item weights
        q.A_ub = p.A_ub * (1.0 + sgn * delta * noise(shape))

    elif ptype == "Portfolio":                   # asset risks
        risk = p.A_ub[0:1, :] * (1.0 + delta * noise((1, p.n)))
        q.A_ub = np.vstack([risk, np.ones((1, p.n))])   # risk' and cash: sum(y) <= 1
        q.b_ub = np.array([p.b_ub[0], 1.0])
        q.A_eq = np.zeros((0, p.n))
        q.b_eq = np.array([])

    elif ptype == "Diet":                        # nutrient content (A_ub = -A)
        s = -1.0 if mode == "adverse" else 1.0   # adverse: content shrinks
        q.A_ub = p.A_ub * (1.0 + s * delta * noise(shape))

    elif ptype == "SetCover":                    # coverage degradation 1 -> 0
        A = -p.A_ub.copy()                       # (n_elem, n_sets) 0/1
        drop = (rng.random(A.shape) < delta) & (A > 0)
        A2 = A * (~drop)
        for i in range(A2.shape[0]):             # keep each element coverable
            if A2[i].sum() == 0:
                j = int(np.flatnonzero(A[i])[rng.integers(int(A[i].sum()))])
                A2[i, j] = 1.0
        q.A_ub = -A2

    elif ptype == "TSP":                         # edge lengths
        c = p.c.copy()
        E = int(p.integrality.sum())
        c[:E] = c[:E] * (1.0 + delta * noise(E))
        q.c = c

    elif ptype == "Scheduling":                  # processing times
        A = p.A_ub.copy()
        m = int(A.shape[0])
        nj = (p.n // m) if p.n % m == 0 else ((p.n - 1) // m)
        for j in range(m):
            for i in range(nj):
                A[j, i * m + j] *= (1.0 + delta * noise(1)[0])
        q.A_ub = A

    elif ptype == "Facility":                    # costs
        q.c = p.c * (1.0 + delta * noise(p.n))

    else:
        raise ValueError(ptype)
    return q


# --------------------------------------------------------------- scoring
def realized_value(p, q, ptype, x):
    """f_realized(R(x)): realized objective after one-sided repair, own sense."""
    if ptype == "Knapsack":                      # may drop items (y <= x)
        r = milp(-p.c, constraints=[LinearConstraint(q.A_ub, -INF, q.b_ub)],
                 integrality=p.integrality, bounds=Bounds(np.zeros(p.n), x),
                 options={"time_limit": 30, "mip_rel_gap": 0})
        return -float(r.fun) if r.success else np.nan

    if ptype == "Portfolio":                     # may sell down into cash (y <= x)
        r = milp(-p.c, constraints=[LinearConstraint(q.A_ub, -INF, q.b_ub)],
                 integrality=np.zeros(p.n, int), bounds=Bounds(np.zeros(p.n), x),
                 options={"time_limit": 30})
        return -float(r.fun) if r.success else np.nan

    if ptype == "Diet":                          # may add amounts (y >= x)
        r = milp(p.c, constraints=[LinearConstraint(q.A_ub, -INF, q.b_ub)],
                 integrality=np.zeros(p.n, int), bounds=Bounds(x, np.full(p.n, INF)),
                 options={"time_limit": 30})
        return float(r.fun) if r.success else np.nan

    if ptype == "SetCover":                      # may add sets (y >= x)
        r = milp(p.c, constraints=[LinearConstraint(q.A_ub, -INF, q.b_ub)],
                 integrality=p.integrality, bounds=Bounds(x, np.ones(p.n)),
                 options={"time_limit": 30, "mip_rel_gap": 0})
        return float(r.fun) if r.success else np.nan

    if ptype == "Scheduling":                    # always feasible: recompute makespan
        m = int(q.A_ub.shape[0])
        nj = (p.n - 1) // m
        assign = x[:nj * m].reshape(nj, m)
        pmat = np.zeros((nj, m))
        for j in range(m):
            for i in range(nj):
                pmat[i, j] = q.A_ub[j, i * m + j]
        return float((assign * pmat).sum(axis=0).max())

    if ptype == "TSP":                           # always feasible: recompute length
        return float(q.c @ x)

    if ptype == "Facility":                      # always feasible: recompute cost
        return float(q.c @ x)

    raise ValueError(ptype)


def regret_of(p, q, ptype, x, f_opt):
    f = realized_value(p, q, ptype, x)
    if not np.isfinite(f) or not np.isfinite(f_opt) or abs(f_opt) < 1e-12:
        return np.nan
    loss = f - f_opt if p.sense == "min" else f_opt - f
    return max(0.0, float(loss / abs(f_opt)))


# ---------------------------------------------------------------- driver
def run(delta_grid, K, limit, types, mode="symmetric"):
    df = pd.read_csv(CSV)
    df = df[(df["parseable"] == True) & (df["feasible"] == True)]  # noqa: E712
    rows, inst_rows = [], []
    i_done = 0
    t0 = time.time()

    for (ptype, inst), g in df.groupby(["problem_type", "instance"], sort=False):
        if ptype not in types:
            continue
        i_done += 1
        if limit and i_done > limit:
            break
        p = get_problem(ptype, inst)

        # unique candidate solutions
        xs, seen = [], {}
        for _, r in g.iterrows():
            x = np.array(json.loads(r["x_json"]), float)
            if len(x) != p.n:
                continue
            key = tuple(np.round(x, 6))
            if key in seen:
                continue
            seen[key] = len(xs)
            xs.append((x, float(r["gap"]), float(r["objective"])))
        if len(xs) < 2:
            continue

        # scenarios -> mean regret per candidate
        reg = np.zeros((len(xs), len(delta_grid), K))
        fopts = np.zeros((len(delta_grid), K))
        for di, d in enumerate(delta_grid):
            for k in range(K):
                rng = np.random.default_rng(zlib.crc32(f"{inst}|{di}|{k}|{mode}".encode()))
                q = make_scenario(p, ptype, d, rng, mode)
                fo = solve_opt(q)
                fopts[di, k] = fo
                for ci, (x, _, _) in enumerate(xs):
                    reg[ci, di, k] = regret_of(p, q, ptype, x, fo)
        mean_reg = np.nanmean(reg, axis=(1, 2))
        worst_reg = np.nanmax(reg, axis=(1, 2))

        gaps = np.array([g_[1] for g_ in xs])
        objs = np.array([g_[2] for g_ in xs])
        ok = np.isfinite(mean_reg)
        if ok.sum() < 2:
            continue

        g_ok, r_ok, o_ok = gaps[ok], mean_reg[ok], objs[ok]
        rho = spearmanr(g_ok, r_ok).statistic
        i_gap, i_reg = int(np.argmin(g_ok)), int(np.argmin(r_ok))
        flip = i_gap != i_reg
        ratio = (r_ok[i_gap] + 1e-12) / (r_ok[i_reg] + 1e-12)
        inst_rows.append({
            "problem_type": ptype, "instance": inst, "n_cand_unique": int(ok.sum()),
            "n_cand_raw": int(len(g)), "n_scen": int(len(delta_grid) * K),
            "rho_gap_regret": rho, "top1_flip": bool(flip),
            "regret_min_gap": float(r_ok[i_gap]), "regret_min_regret": float(r_ok[i_reg]),
            "regret_ratio": float(ratio),
            "regret_median": float(np.median(r_ok)),
            "gap_of_min_gap": float(g_ok[i_gap]), "gap_of_min_regret": float(g_ok[i_reg]),
            "argmin_gap_above_median": bool(r_ok[i_gap] > np.median(r_ok)),
            "fopt_mean": float(np.nanmean(fopts)), "sense": p.sense,
        })
        for ci, (x, gp, ob) in enumerate(xs):
            rows.append({"problem_type": ptype, "instance": inst, "cand": ci,
                         "name": inst, "gap": gp, "objective": ob,
                         "mean_regret": float(mean_reg[ci]),
                         "worst_regret": float(worst_reg[ci]),
                         "is_argmin_gap": bool(ci == np.flatnonzero(ok)[i_gap]) if ok[ci] else False,
                         "is_argmin_regret": bool(ci == np.flatnonzero(ok)[i_reg]) if ok[ci] else False})
        if i_done % 10 == 0:
            print(f"  [{i_done}] {inst} uniq={ok.sum()} rho={rho:.2f} flip={flip} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    cand_df = pd.DataFrame(rows)
    inst_df = pd.DataFrame(inst_rows)
    tag = mode
    cand_df.to_csv(os.path.join(OUT, f"regret_pilot_candidates_{tag}.csv"), index=False)
    inst_df.to_csv(os.path.join(OUT, f"regret_pilot_instances_{tag}.csv"), index=False)

    summ = {}
    for pt, gI in inst_df.groupby("problem_type"):
        summ[pt] = {
            "n_instances": int(len(gI)),
            "flip_rate": float(gI["top1_flip"].mean()),
            "median_rho": float(gI["rho_gap_regret"].median()),
            "mean_rho": float(gI["rho_gap_regret"].mean()),
            "median_regret_ratio": float(gI["regret_ratio"].median()),
            "share_argmin_gap_above_median": float(gI["argmin_gap_above_median"].mean()),
            "median_regret_min_gap": float(gI["regret_min_gap"].median()),
        }
    summ["_overall"] = {
        "n_instances": int(len(inst_df)),
        "flip_rate": float(inst_df["top1_flip"].mean()),
        "median_rho": float(inst_df["rho_gap_regret"].median()),
        "share_argmin_gap_above_median": float(inst_df["argmin_gap_above_median"].mean()),
        "median_regret_ratio": float(inst_df["regret_ratio"].median()),
        "delta_grid": list(delta_grid), "K": K, "mode": mode,
    }
    with open(os.path.join(OUT, f"regret_pilot_summary_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    return inst_df, summ


def validate(limit=0):
    df = pd.read_csv(CSV)
    diffs = []
    n = 0
    for (ptype, inst), g in df.groupby(["problem_type", "instance"], sort=False):
        if limit and n >= limit:
            break
        n += 1
        p = get_problem(ptype, inst)
        R.solve_problem(p)
        o = float(g["opt"].iloc[0])
        diffs.append((inst, ptype, p.opt, o, abs(p.opt - o) / max(abs(o), 1e-9)))
        if n % 40 == 0:
            print(f"  checked {n}", flush=True)
    d = pd.DataFrame(diffs, columns=["instance", "problem_type", "opt_rebuilt", "opt_csv", "rel_err"])
    print(d.groupby("problem_type")["rel_err"].max())
    print("max rel err:", d["rel_err"].max())
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run", choices=["validate", "run"])
    ap.add_argument("--deltas", default="0.05,0.10,0.20")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--types", default="Knapsack,Portfolio,Diet,SetCover,TSP,Scheduling,Facility")
    ap.add_argument("--mode", default="symmetric", choices=["symmetric", "adverse"])
    a = ap.parse_args()
    if a.stage == "validate":
        validate(a.limit)
    else:
        deltas = [float(s) for s in a.deltas.split(",")]
        inst_df, summ = run(deltas, a.k, a.limit, set(a.types.split(",")), a.mode)
        print(json.dumps(summ, ensure_ascii=False, indent=2))
