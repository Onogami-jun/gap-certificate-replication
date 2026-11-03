#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pairwise (like-for-like) release test for the two-dimensional rule.

Reads the archived plan-level ranks (screen_plans.csv) and the archived
candidate tables, and writes screen_pairwise.csv:

  * the plans the exposure quantile removes vs the plans it keeps, with
    false-release counts, risk ratio and Fisher exact p (Table 6, text of
    Sec. 5.6);
  * the per-instance chance level 1 - 1/n of the top-1 conflict, so that the
    conflict rate can be read against its own null (Sec. 5.2).

No random seed is used: the test is exact (hypergeometric).
"""
import csv
import io
import itertools
import math
import os
import zipfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tre_replication.zip")
OUT = os.path.join(HERE, "screen_pairwise.csv")

TOP_RISK = 0.75          # top quartile of the within-instance risk rank
EXPO_CUT = 0.50          # the second dimension is fixed at 0.5 (Table 6)
TAUS = (0.50, 0.25)


def read_zip(name):
    """Read a sibling CSV (inside the replication pack) or from the pack itself."""
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    with zipfile.ZipFile(SRC) as z:
        return list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8"))))


def fisher_two_sided(a, b, c, d):
    """Exact Fisher test on [[a, b], [c, d]]; a = dropped & wrong."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x):
        return (math.comb(row1, x) * math.comb(n - row1, col1 - x)
                / math.comb(n, col1))

    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    p0 = prob(a)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p0 * 1 + 1e-12)


def pair_rows():
    rows = read_zip("screen_plans.csv")
    out = []
    for fam in ("C", "D"):
        pool = [r for r in rows if r["family"] == fam]
        n_plans = len(pool)
        for tau in TAUS:
            admitted = [r for r in pool if float(r["gap_rank"]) <= tau + 1e-9]
            kept = [r for r in admitted if float(r["expo_rank"]) <= EXPO_CUT + 1e-9]
            dropped = [r for r in admitted if float(r["expo_rank"]) > EXPO_CUT + 1e-9]
            kw = sum(1 for r in kept if float(r["risk_rank"]) >= TOP_RISK - 1e-9)
            dw = sum(1 for r in dropped if float(r["risk_rank"]) >= TOP_RISK - 1e-9)
            rr = (kw / len(kept)) / (dw / len(dropped)) if kept and dropped else float("nan")
            p = fisher_two_sided(dw, len(dropped) - dw, kw, len(kept) - kw)
            out.append(dict(family=fam, tau=tau, n_plans=n_plans,
                            gap_admitted=len(admitted),
                            dropped=len(dropped), dropped_wrong=dw,
                            dropped_rate=round(dw / len(dropped), 6),
                            kept=len(kept), kept_wrong=kw,
                            kept_rate=round(kw / len(kept), 6),
                            risk_ratio=round(rr, 4), fisher_p=round(p, 6)))
    return out


def null_rows():
    """Per-instance chance level 1 - 1/n of the top-1 conflict."""
    sources = {
        "C": ("llm_slot_candidates.csv", "inst"),
        "D": ("llm_bunker_candidates.csv", "inst"),
        "B": ("regret_pilot_candidates.csv", "inst"),
    }
    out = []
    for fam, (name, key) in sources.items():
        try:
            rows = read_zip(name)
        except KeyError:
            continue
        cols = rows[0].keys()
        if key not in cols:
            continue
        pools = defaultdict(int)
        for r in rows:
            if "feasible" in cols and r["feasible"].strip().lower() not in ("true", "1"):
                continue
            pools[r[key]] += 1
        n = [v for v in pools.values() if v >= 2]
        if not n:
            continue
        nulls = [1 - 1 / k for k in n]
        out.append(dict(family=fam, n_instances=len(n),
                        median_n=sorted(n)[len(n) // 2],
                        min_n=min(n), max_n=max(n),
                        mean_null=round(sum(nulls) / len(nulls), 4)))
    return out


if __name__ == "__main__":
    pairs = pair_rows()
    nulls = null_rows()
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)
    print("pairs ->", OUT)
    for r in pairs:
        print("  {family} tau={tau}  dropped {dropped_wrong}/{dropped} ({dropped_rate}) "
              "vs kept {kept_wrong}/{kept} ({kept_rate})  RR={risk_ratio}  p={fisher_p}".format(**r))
    for r in nulls:
        print("  null {family}: n_inst={n_instances} n_med={median_n} "
              "range=[{min_n},{max_n}] mean 1-1/n={mean_null}".format(**r))
