#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
案例集合版（回应 referee R5.2：N=1 不能叫 construct validity）。
20 个合成订舱簿 × 同一真实航线/参数，每册 18 个启发式与保守方案；前 5 册另加 4 个 LLM 方案。
计划时点即超舱的方案按 C 族协议剔除（不再进入 gap 排序），单独计数。
输出 case_ensemble.csv / case_ensemble_summary.json
"""
import json, os, sys, zlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(OUT, "real_case.py"), encoding="utf-8").read().split('if __name__ == "__main__"')[0]
g = {"__name__": "case_lib", "__file__": os.path.join(OUT, "real_case.py")}
exec(src, g)

gen_book, joint_opt, greedy_slots = g["gen_book"], g["joint_opt"], g["greedy_slots"]
fuel_draft_penalty, risk_of = g["fuel_draft_penalty"], g["risk_of"]
B_TOTAL, B_SG_LEG, TANK, CAP_TEU = g["B_TOTAL"], g["B_SG_LEG"], g["TANK"], g["CAP_TEU"]
VLSFO, build_prompt, parse, call_ollama = g["VLSFO"], g["build_prompt"], g["parse"], g["call_ollama"]

USE_LLM = "--no-llm" not in sys.argv
N_BOOKS, N_LLM_BOOKS = 20, 5
grid_sg = np.linspace(B_SG_LEG, TANK["Singapore"], 9)
grid_rt = np.linspace(0.0, TANK["Rotterdam"], 7)
bunkers = {"min-lift": (B_SG_LEG, TANK["Rotterdam"]),
           "balanced": (0.5 * (B_SG_LEG + TANK["Singapore"]), 400.0),
           "tank-full": (TANK["Singapore"], 0.0)}

rows = []
for si in range(N_BOOKS):
    seed = si + 1
    book = gen_book(seed)
    opt, xopt, ysg0, yrt0 = joint_opt(book, grid_sg, grid_rt)
    plans = []
    for tag, (ysg, yrt) in bunkers.items():
        pen = fuel_draft_penalty(ysg, yrt)
        for kname, key in [("rate", book["r"] * book["q"]),
                           ("density", book["r"] * book["q"] / np.maximum(book["w"] @ CAP_TEU, 1))]:
            plans.append({"src": f"heur:{kname}/{tag}", "y_sg": ysg, "y_rt": yrt,
                          "x": greedy_slots(book, key, penalty=pen)})
    for frac in (0.95, 0.90, 0.80):
        ysg, yrt = bunkers["balanced"]
        plans.append({"src": f"slack:{frac:.0%}", "y_sg": ysg, "y_rt": yrt,
                      "x": greedy_slots(book, book["r"] * book["q"],
                                        penalty=fuel_draft_penalty(ysg, yrt), fill_frac=frac)})
    if USE_LLM and si < N_LLM_BOOKS:
        prompt = build_prompt(book)
        for model in ("qwen2.5:7b", "mistral:7b"):
            for t in (0.4, 1.0):
                sd = int(zlib.crc32(f"{seed}|{model}|{t}".encode()) % 1_000_000)
                pr = parse(call_ollama(model, prompt, t, sd), book)
                if pr is None:
                    continue
                x, ysg, yrt = pr
                if ysg + yrt < B_TOTAL - 1e-6:
                    ysg = min(TANK["Singapore"], ysg + (B_TOTAL - ysg - yrt))
                plans.append({"src": f"llm:{model}@{t}", "x": x, "y_sg": ysg, "y_rt": yrt})

    recs, n_infeas = [], 0
    for p in plans:
        rev = float(book["r"] * book["q"] @ p["x"])
        obj = rev - (VLSFO["Singapore"] * p["y_sg"] + VLSFO["Rotterdam"] * p["y_rt"])
        load = (book["w"] * (book["q"] * p["x"])[:, None]).sum(0)
        if np.any(load > CAP_TEU + 1e-9):
            n_infeas += 1
            continue
        risk, bump, short, p95 = risk_of(book, p["x"], p["y_sg"], p["y_rt"])
        recs.append({"book": seed, "plan": p["src"], "gap": (opt - obj) / abs(opt),
                     "risk": risk, "p95": p95, "teu": float(book["q"] @ p["x"]),
                     "util": float(np.max(load / CAP_TEU))})
    R = pd.DataFrame(recs)
    if len(R) < 3 or np.ptp(R.risk) == 0:
        continue
    R["n_infeasible"] = n_infeas
    rows.append(R)
    ig, ir = int(np.argmin(R.gap.values)), int(np.argmin(R.risk.values))
    print(f"book {seed:2d}: n={len(R):2d} infeas={n_infeas} rho={spearmanr(R.gap,R.risk).statistic:+.3f} "
          f"conflict={'YES' if ig!=ir else 'NO '} gap*={R.gap.values[ig]:.4f}/{R.gap.values[ir]:.4f} "
          f"risk*={R.risk.values[ig]:.3f}/{R.risk.values[ir]:.3f}", flush=True)

E = pd.concat(rows, ignore_index=True)
E.to_csv(os.path.join(OUT, "case_ensemble.csv"), index=False)
per = []
for b, gg in E.groupby("book"):
    ig, ir = int(np.argmin(gg.gap.values)), int(np.argmin(gg.risk.values))
    per.append({"book": b, "n": len(gg), "rho": spearmanr(gg.gap, gg.risk).statistic,
                "conflict": bool(ig != ir),
                "risk_mingap": gg.risk.values[ig], "risk_minrisk": gg.risk.values[ir],
                "gap_minrisk": gg.gap.values[ir], "gap_mingap": gg.gap.values[ig]})
PP = pd.DataFrame(per)
PP.to_csv(os.path.join(OUT, "case_ensemble_books.csv"), index=False)
S = {"books": len(PP), "conflict_rate": float(PP.conflict.mean()),
     "rho_med": float(PP.rho.median()),
     "risk_mingap_med": float(PP.risk_mingap.median()),
     "risk_minrisk_med": float(PP.risk_minrisk.median()),
     "gap_mingap_med": float(PP.gap_mingap.median()),
     "gap_minrisk_med": float(PP.gap_minrisk.median()),
     "plans_per_book_med": float(PP.n.median()),
     "infeasible_plans": int(E.n_infeasible.fillna(0).groupby(E.book).max().sum())}
json.dump(S, open(os.path.join(OUT, "case_ensemble_summary.json"), "w", encoding="utf-8"), indent=2)
print("\n=== 汇总 ===")
for k, v in S.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
print("saved case_ensemble.csv / case_ensemble_books.csv")
