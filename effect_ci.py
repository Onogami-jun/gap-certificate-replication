#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paired exposure penalties with instance-level bootstrap intervals (Table 4).

The conflict rates already carry Wilson and bootstrap intervals (§4.4); the
effect sizes did not, which is the methodological gap a referee is entitled to
press. Every penalty here is the median over instances of
    exposure(gap-optimal plan) - exposure(exposure-optimal plan),
i.e. a paired statistic, with a percentile bootstrap over instances (5,000
resamples, fixed seed). The medians are chosen so that the two exposure columns
reproduce the published values.
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
B = 5000


ALPHA = 0.05 / 9  # Bonferroni over the nine rows of Table 4


def ci(d, rng):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    draws = rng.choice(d, size=(B, d.size), replace=True)
    med = np.median(draws, axis=1)
    return (d.size, float(np.median(d)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5)),
            float(np.percentile(med, 100 * ALPHA / 2)), float(np.percentile(med, 100 * (1 - ALPHA / 2))))


rows = []

p = pd.read_csv(os.path.join(OUT, "prop_instances.csv"))
g = p[p.family.str.startswith("C")]
r = np.random.default_rng(20261028)
n, m, lo, hi, lob, hib = ci(g.risk_x_min_gap - g.risk_x_min_risk, r)
rows.append(dict(row="C LLM adverse", n=n, gap_med=np.median(g.risk_x_min_gap),
                 safe_med=np.median(g.risk_x_min_risk), penalty=m, lo=lo, hi=hi, lo_b=lob, hi_b=hib))

b = pd.read_csv(os.path.join(OUT, "llm_bunker_instances.csv"))
b = b[b.n >= 3]
r = np.random.default_rng(20261028)
n, m, lo, hi, lob, hib = ci(b.ovr_min_gap - b.ovr_min_ovr, r)
rows.append(dict(row="D LLM adverse", n=n, gap_med=np.median(b.ovr_min_gap),
                 safe_med=np.median(b.ovr_min_ovr), penalty=m, lo=lo, hi=hi, lo_b=lob, hi_b=hib))

d = pd.read_csv(os.path.join(OUT, "rev2_drift.csv"))
for fam, regime in [("C", "adverse"), ("C", "symmetric"), ("D", "adverse"),
                    ("D", "symmetric"), ("D", "favourable")]:
    q = d[(d.fam == fam) & (d.regime == regime)]
    r = np.random.default_rng(20261028)
    n, m, lo, hi, lob, hib = ci(q.risk_mingap - q.risk_minrisk, r)
    rows.append(dict(row=f"{fam} heuristic {regime}", n=n, gap_med=np.median(q.risk_mingap),
                     safe_med=np.median(q.risk_minrisk), penalty=m, lo=lo, hi=hi, lo_b=lob, hi_b=hib))

A = pd.read_csv(os.path.join(OUT, "rev2b_books_arms.csv"))
A = A[A.regime == "adverse"]
for arm, name in [("base", "Books base"), ("base+slack", "Books + slack")]:
    q = A[A.arm == arm]
    r = np.random.default_rng(20261028)
    n, m, lo, hi, lob, hib = ci(q.risk_mingap - q.risk_minrisk, r)
    rows.append(dict(row=name, n=n, gap_med=np.median(q.risk_mingap),
                     safe_med=np.median(q.risk_minrisk), penalty=m, lo=lo, hi=hi, lo_b=lob, hi_b=hib))

C = pd.DataFrame(rows)
C.to_csv(os.path.join(OUT, "effect_ci.csv"), index=False)
print(C.round(4).to_string(index=False))
