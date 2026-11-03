#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""表 6 的 Wilson 区间与 25 % 随机基准。

一个随机释放的集合里，落入风险上四分位的比例期望恰为 25 %；这不是"零错误"，
而是任何放行规则必须超过的门槛。释放数只有 13、19、39 个，点估计本身分辨不了
25 % 与 30 % 的差别，所以区间必须一并报告。
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.959964):
    """二项比例的 Wilson 得分区间。"""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


S = pd.read_csv(os.path.join(OUT, "screen_rule.csv"))
rows = []
for _, r in S.iterrows():
    n_gap = int(round(r.acc_rate_gap_only * r.n_plans))
    n_2d = int(round(r.acc_rate_2d * r.n_plans))
    k_gap = int(round(r.false_rel_gap_only * n_gap))
    k_2d = int(round(r.false_rel_2d * n_2d))
    lo_g, hi_g = wilson(k_gap, n_gap)
    if n_2d >= 3:
        lo_2, hi_2 = wilson(k_2d, n_2d)
    else:
        lo_2, hi_2 = np.nan, np.nan          # 1/1 的格子不给估计
    rows.append(dict(family=r.family, tau=r.gap_cut, n_plans=int(r.n_plans),
                     n_rel_gap=n_gap, k_gap=k_gap, fr_gap=k_gap / n_gap,
                     lo_gap=lo_g, hi_gap=hi_g, n_rel_2d=n_2d, k_2d=k_2d,
                     fr_2d=(k_2d / n_2d if n_2d else np.nan),
                     lo_2d=lo_2, hi_2d=hi_2, baseline=0.25,
                     mean_rank_gap=r.mean_riskrank_gap_only,
                     mean_rank_2d=r.mean_riskrank_2d))

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "screen_ci.csv"), index=False)
pd.set_option("display.width", 200)
print(T.round(4).to_string(index=False))
