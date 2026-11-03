#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""表 4 第二列：实例内杠杆离散度（变异系数）的中位数。

定义（与表注一致）：对每个实例，取该实例候选池内杠杆值的变异系数
（标准差 / 均值），再对实例取中位数。C 的杠杆是舱位利用率 util，
D 的杠杆是油舱充满度 fill；启发式两族用同样的口径，样本为各自档案的全部实例。

样本口径与表 2 一致：C 保留至少有 3 个可行方案且溢出量取值不少于 2 个的实例
（24 个），D 保留至少 3 个方案的实例（39 个）。
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
R = lambda f: pd.read_csv(os.path.join(OUT, f))

rows = []

# C：LLM 与启发式
c = R("llm_slot_candidates.csv")
c = c[c.feasible]
ok = c.groupby("inst").apply(lambda g: len(g) >= 3 and g.overflow.nunique() >= 2,
                             include_groups=False)
cv = c[c.inst.isin(ok[ok].index)].groupby("inst").apply(
    lambda g: g.util.std(ddof=1) / g.util.mean(), include_groups=False)
rows.append(dict(family="C", pool="LLM", n=len(cv), cv_median=float(cv.median())))

h = R("sea_slot_candidates.csv")
cv = h.groupby("inst").apply(lambda g: g.util.std(ddof=1) / g.util.mean(),
                             include_groups=False)
rows.append(dict(family="C", pool="heuristic", n=len(cv), cv_median=float(cv.median())))

# D：LLM 与启发式
b = R("llm_bunker_candidates.csv")
ok = b.groupby("inst").size() >= 3
cv = b[b.inst.isin(ok[ok].index)].groupby("inst").apply(
    lambda g: g.fill.std(ddof=1) / g.fill.mean(), include_groups=False)
rows.append(dict(family="D", pool="LLM", n=len(cv), cv_median=float(cv.median())))

hb = R("sea_bunker_candidates.csv")
cv = hb.groupby("inst").apply(lambda g: g.fill.std(ddof=1) / g.fill.mean(),
                              include_groups=False)
rows.append(dict(family="D", pool="heuristic", n=len(cv), cv_median=float(cv.median())))

L = pd.DataFrame(rows)
L.to_csv(os.path.join(OUT, "lever_cv.csv"), index=False)
print(L.round(4).to_string(index=False))
