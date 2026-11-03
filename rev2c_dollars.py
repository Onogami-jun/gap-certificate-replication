# -*- coding: utf-8 -*-
"""Dollar denominators for every lift-hedging policy in family D.

Writes rev2c_dollars.csv with, per instance and policy, the committed cost
(the denominator an operator would quote) and the absolute exposure in USD.
"""
import os
import zlib

import numpy as np
import pandas as pd

import rev2_experiments as R  # noqa: F401  (reuses the loaded generators)

BK = R.BK
OUT = os.path.dirname(os.path.abspath(__file__))

rows = []
for s in range(40):
    inst = BK["gen"](2000 + s)
    opt, zo, yo = BK["solve"](inst)
    if not np.isfinite(opt):
        continue
    r0, ab0 = R.bunker_risk(inst, zo, yo)
    rows.append({"fam": "D", "policy": "nominal-opt", "committed": opt,
                 "risk": r0, "exposure_abs": ab0})
    for q in (0.50, 0.75, 0.90, 0.95):
        scale = 1 + 0.20 * q
        y2 = np.minimum(inst["cap"], yo * scale)
        if y2.sum() < inst["B"]:
            continue
        z2 = (y2 > 1e-9).astype(int)
        cost2 = BK["eval_cost"](inst, z2, y2)
        rk, ab = R.bunker_risk(inst, z2, y2)
        rows.append({"fam": "D", "policy": f"hedge-q{int(q*100)}", "committed": cost2,
                     "risk": rk, "exposure_abs": ab})

DL = pd.DataFrame(rows)
DL.to_csv(os.path.join(OUT, "rev2c_dollars.csv"), index=False)
print(DL.groupby("policy").agg(n=("committed", "size"), committed=("committed", "median"),
                               risk=("risk", "median"), exposure=("exposure_abs", "median"),
                               risk_mean=("risk", "mean")).round(3).to_string())
