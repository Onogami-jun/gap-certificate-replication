#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""表 5 中 D 族的美元行：gap 最优方案与最稳方案。

表 5 的美元列原本只有名义最优与四个对冲策略有归档脚本（rev2c_dollars.py）；
gap 最优与最稳两行是直接从候选档案取得，从未落盘，审稿人无法回溯。
本脚本按同一定义（对每个实例取该策略的承诺成本与绝对暴露，再对实例取中位）
把这两行写出来，样本与表 2/表 4 一致：至少 3 个方案的 D 实例（39 个）。
"""
import os

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
B = pd.read_csv(os.path.join(OUT, "llm_bunker_candidates.csv"))
keep = B.groupby("inst").size()
B = B[B.inst.isin(keep[keep >= 3].index)]

rows = []
for policy, idx in [("gap-optimal plan", B.groupby("inst").gap.idxmin()),
                    ("safest plan", B.groupby("inst").overrun.idxmin())]:
    q = B.loc[idx]
    rows.append(dict(policy=policy, n=len(q),
                     committed=float(q.cost.median()),
                     exposure_usd=float(q.abs_overrun.median()),
                     gap=float(q.gap.median()),
                     exposure_ratio=float(q.overrun.median())))

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "table5_dollars.csv"), index=False)
print(T.round(4).to_string(index=False))
