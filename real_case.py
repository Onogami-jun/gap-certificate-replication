#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4.5 真实案例（小规模）：一条亚欧航线、一个航次的"舱位 + 加油"联合承诺方案。

诚实标注：
  * 航线结构、港口顺序、航段距离、船型、2024 年量级的 VLSFO 价格与港口费按公开资料标定；
  * 订舱簿（60 个 request 的 OD / 箱量 / 运价）为合成数据 —— 真实订舱明细不公开；
  * 因此这是一个"参数按真实量级标定、订舱数据合成"的案例，用于展示机制，不用于估计总体效应。

模型（一次航次）：
  港口 0 Shanghai -> 1 Ningbo -> 2 Singapore -> 3 Rotterdam -> 4 Hamburg
  航段 k 的舱位上限 cap_k（Suez/Malacca/Malacca 吃水限制 + Hamburg 易北河吃水限制）
  决策 x_i ∈{0,1} 接受哪些订舱；y_SG, y_RT 在新加坡/鹿特丹的加油量
  目标 max  Σ r_i q_i x_i − (p_SG y_SG + p_RT y_RT)
  约束 每航段 Σ_{i覆盖k} q_i x_i ≤ cap_k − 富余燃油重量/11(t/TEU)
        y_SG ≥ 新加坡—鹿特丹航段需油；y_SG ≤ 4200；y_RT ≤ 1200；y_SG+y_RT ≥ B

承诺制：订舱与燃油条款在开航前锁定。
漂移：箱量上浮、油价上浮、（吃水/天气/空箱占位导致的）有效舱位下降、实际耗油上浮。
违约代价：甩柜按 30% 运价赔付 + 缺口油按 1.5× 现货溢价补。
"""
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------- 真实量级参数 ----------------
PORTS = ["Shanghai", "Ningbo", "Singapore", "Rotterdam", "Hamburg"]
LEG_NM = np.array([185.0, 2120.0, 8100.0, 310.0])
CAP_TEU = np.array([24000.0, 24000.0, 22800.0, 20400.0])   # 最后一段受易北河吃水限制
PORT_FEE = np.array([45000.0, 28000.0, 32000.0, 41000.0, 47000.0])
VLSFO = {"Singapore": 612.0, "Rotterdam": 588.0}           # USD/t, 2024 量级
TANK = {"Singapore": 4200.0, "Rotterdam": 1200.0}
SPEED_KN, T_PER_DAY = 18.5, 135.0
SEA_MARGIN = 1.12
B_TOTAL = float(LEG_NM.sum() / (SPEED_KN * 24.0) * T_PER_DAY * SEA_MARGIN)   # ≈ 3.6 kt
B_SG_LEG = float(LEG_NM[2] / (SPEED_KN * 24.0) * T_PER_DAY * 1.05)            # 新加坡最少加油量
T_PER_TEU = 11.0                                          # 吃水置换：1 TEU ≈ 11 t
BUMP_PENALTY, SPOT_PREMIUM = 0.30, 1.50
N_BOOK = 60


def gen_book(seed=7):
    """合成订舱簿：OD 对、箱量、运价。
    运价按"每 TEU-海里 0.20 USD"标定（亚欧即 ~2,000 USD/TEU，与 2024 年均现货运价同量级），
    再乘对数正态扰动，使"高运价/远航段"与"低运价/短航段"形成真实取舍。"""
    rng = np.random.default_rng(seed)
    legs = []
    for _ in range(N_BOOK):
        o = int(rng.integers(0, 4))
        d = int(rng.integers(o + 1, 5))
        legs.append((o, d))
    q = rng.uniform(100, 1200, N_BOOK)
    dist = np.array([LEG_NM[o:d].sum() for o, d in legs])
    r = 0.20 * dist * np.exp(0.25 * rng.standard_normal(N_BOOK))
    w = np.zeros((N_BOOK, 4))
    for i, (o, d) in enumerate(legs):
        w[i, o:d] = 1.0
    # 各航段需求/舱位比不同：Suez 段最紧，Elbe 段次之
    tight = np.array([1.02, 1.05, 1.22, 1.10])
    load = (w * q[:, None]).sum(axis=0)
    q = q * float(np.min(tight * CAP_TEU / load))
    return dict(o=np.array([l[0] for l in legs]), d=np.array([l[1] for l in legs]),
                q=q, r=r, w=w, n=N_BOOK)


def fuel_draft_penalty(y_sg, y_rt):
    """Hamburg 段的富余燃油重量折算成少装的 TEU（吃水限制）。"""
    remaining = max(0.0, y_sg + y_rt - (B_TOTAL - LEG_NM[3] / (SPEED_KN * 24) * T_PER_DAY))
    return remaining / T_PER_TEU


def slot_milp(book, cap_extra_penalty=0.0):
    """给定燃油吃水惩罚，求舱位分配最优（多维背包）。"""
    caps = CAP_TEU.copy()
    caps[3] = max(1000.0, caps[3] - cap_extra_penalty)
    A = (book["w"] * book["q"][:, None] / caps[None, :]).T
    res = milp(c=-book["r"] * book["q"], constraints=LinearConstraint(A, -np.inf, np.ones(4)),
               integrality=np.ones(book["n"]), bounds=Bounds(0, 1))
    if not res.success:
        return np.nan, None
    return float(book["r"] * book["q"] @ np.round(res.x)), np.round(res.x).astype(int)


def joint_opt(book, grid_sg, grid_rt):
    best = (-np.inf, None, None, None)
    for ysg in grid_sg:
        for yrt in grid_rt:
            if ysg + yrt < B_TOTAL - 1e-6 or ysg < B_SG_LEG - 1e-6:
                continue
            rev, x = slot_milp(book, fuel_draft_penalty(ysg, yrt))
            if not np.isfinite(rev):
                continue
            obj = rev - (VLSFO["Singapore"] * ysg + VLSFO["Rotterdam"] * yrt)
            if obj > best[0]:
                best = (obj, x, ysg, yrt)
    return best


def greedy_slots(book, key, rng=None, noise=0.0, penalty=0.0, fill_frac=1.0):
    n = book["n"]
    k = key.copy()
    if noise and rng is not None:
        k = k * (1 + noise * rng.standard_normal(n))
    order = np.argsort(-k)
    load = np.zeros(4)
    caps = CAP_TEU * fill_frac
    caps[3] = max(1000.0, caps[3] - penalty)
    x = np.zeros(n)
    for i in order:
        if np.all(load + book["w"][i] * book["q"][i] <= caps + 1e-9):
            x[i] = 1
            load += book["w"][i] * book["q"][i]
    return x


def local_search(book, x, rng, penalty=0.0):
    cap = CAP_TEU.copy()
    cap[3] = max(1000.0, cap[3] - penalty)
    for _ in range(4):
        imp = False
        for i in np.flatnonzero(x == 1):
            for j in np.flatnonzero(x == 0):
                y = x.copy()
                y[i], y[j] = 0, 1
                if np.all((book["w"] * (book["q"] * y)[:, None]).sum(0) <= cap + 1e-9) and \
                        book["r"] * book["q"] @ y > book["r"] * book["q"] @ x + 1e-9:
                    x, imp = y, True
                    break
            if imp:
                break
    return x


def risk_of(book, x, y_sg, y_rt, n_scen=200):
    """承诺制下的实现风险：超过承诺预算的幅度（相对承诺毛利）。"""
    rev = float(book["r"] * book["q"] @ x)
    fees = float(PORT_FEE.sum())
    committed_cost = VLSFO["Singapore"] * y_sg + VLSFO["Rotterdam"] * y_rt + fees
    committed_margin = rev - committed_cost
    out, bumped, shorts = [], [], []
    for s in range(n_scen):
        dl = 0.10 if s < n_scen // 2 else 0.20
        rng = np.random.default_rng(zlib.crc32(f"{s}|{y_sg:.1f}|{y_rt:.1f}".encode()))
        q2 = book["q"] * (1 + dl * rng.uniform(0, 1, book["n"]))
        caps2 = CAP_TEU * (1 - dl * rng.uniform(0, 1, 4))
        caps2[3] = max(1000.0, caps2[3] - fuel_draft_penalty(y_sg, y_rt))
        p2 = {k: v * (1 + dl * rng.uniform(0, 1)) for k, v in VLSFO.items()}
        b2 = B_TOTAL * (1 + dl * rng.uniform(0, 1))
        load = (book["w"] * (q2 * x)[:, None]).sum(0)
        over = np.maximum(0.0, load - caps2)
        bump_frac = float(min(1.0, over.max() / max(1e-9, load.max())))
        lost = bump_frac * rev
        got = min(y_sg, TANK["Singapore"]) + min(y_rt, TANK["Rotterdam"])
        short = max(0.0, b2 - got)
        fuel = p2["Singapore"] * min(y_sg, TANK["Singapore"]) + \
            p2["Rotterdam"] * min(y_rt, TANK["Rotterdam"]) + short * SPOT_PREMIUM * max(p2.values())
        realized = (rev - lost) - fuel - fees - BUMP_PENALTY * lost
        out.append((committed_margin - realized) / abs(committed_margin))
        bumped.append(bump_frac)
        shorts.append(short / B_TOTAL)
    return (float(np.mean(out)), float(np.mean(bumped)), float(np.mean(shorts)),
            float(np.percentile(out, 95)))


def call_ollama(model, prompt, temp, seed, timeout=180):
    import urllib.request
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": temp, "seed": seed, "num_predict": 320}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())["response"]
    except Exception:
        return None


def build_prompt(book):
    rows = [f'{i}: leg {book["o"][i]}->{book["d"][i]-1}, qty={book["q"][i]:.0f} TEU, '
            f'rate={book["r"][i]:.0f} USD/TEU' for i in range(book["n"])]
    return ("You are the voyage planner of a 24,000 TEU container ship on the rotation "
            "Shanghai(0)->Ningbo(1)->Singapore(2)->Rotterdam(3)->Hamburg(4).\n"
            "Leg capacities (TEU) are 24000, 24000, 22800, 20400; each leg 3->4 loses one TEU "
            "for every 11 tonnes of fuel still on board (Elbe draft limit).\n"
            "Fuel: VLSFO Singapore 612 USD/t (tank 4200 t), Rotterdam 588 USD/t (tank 1200 t). "
            f"Voyage need {B_TOTAL:.0f} t; at least {B_SG_LEG:.0f} t must be lifted at Singapore "
            "to reach Rotterdam.\n\nBooking requests:\n" + "\n".join(rows) +
            "\n\nChoose the bookings to accept and the bunker lift at Singapore and Rotterdam "
            "to maximise voyage contribution (revenue minus fuel).\n"
            'Output ONLY JSON: {"accept": [ids], "bunker": {"Singapore": tonnes, "Rotterdam": tonnes}}')


def parse(text, book):
    if not text:
        return None
    try:
        o = json.loads(text)
    except Exception:
        return None
    ids = o.get("accept")
    if not isinstance(ids, list):
        return None
    x = np.zeros(book["n"])
    for i in ids:
        try:
            j = int(i)
        except Exception:
            continue
        if 0 <= j < book["n"]:
            x[j] = 1
    bk = o.get("bunker") or {}
    def pick(d, name, default):
        for k, v in d.items():
            if str(k).strip().lower() == name.lower():
                try:
                    return float(v)
                except Exception:
                    return default
        return default
    ysg = min(TANK["Singapore"], max(0.0, pick(bk, "Singapore", B_SG_LEG)))
    yrt = min(TANK["Rotterdam"], max(0.0, pick(bk, "Rotterdam", 0.0)))
    return x, ysg, yrt


if __name__ == "__main__":
    book = gen_book()
    grid_sg = np.linspace(B_SG_LEG, TANK["Singapore"], 9)
    grid_rt = np.linspace(0.0, TANK["Rotterdam"], 7)
    opt, xopt, ysg0, yrt0 = joint_opt(book, grid_sg, grid_rt)
    print(f"B={B_TOTAL:.0f} t, Singapore 段最少 {B_SG_LEG:.0f} t; 联合最优 obj={opt:,.0f} "
          f"(x*={int(xopt.sum())} 票, SG={ysg0:.0f} t, RT={yrt0:.0f} t)")

    # --- 候选 1：启发式舱位 × 三档加油策略（舱位按各自吃水惩罚重新贪心）---
    rng = np.random.default_rng(11)
    bunkers = {"min-lift": (B_SG_LEG, TANK["Rotterdam"]),
               "balanced": (0.5 * (B_SG_LEG + TANK["Singapore"]), 400.0),
               "tank-full": (TANK["Singapore"], 0.0)}
    plans = []
    for tag, (ysg, yrt) in bunkers.items():
        pen = fuel_draft_penalty(ysg, yrt)
        cand = {"rate": greedy_slots(book, book["r"] * book["q"], penalty=pen),
                "density": greedy_slots(book, book["r"], penalty=pen),
                "noise": greedy_slots(book, book["r"], rng, 0.35, pen)}
        cand["ls"] = local_search(book, greedy_slots(book, book["r"] * book["q"], penalty=pen), rng, pen)
        for nm, x in cand.items():
            plans.append({"src": f"heur:{nm}/{tag}", "x": x, "y_sg": ysg, "y_rt": yrt})
    # 保守方案：刻意留出舱位余量（95% / 90% / 80%），这是实际操作中唯一能对冲箱量上浮的做法
    for frac in (0.95, 0.90, 0.80):
        ysg, yrt = bunkers["balanced"]
        plans.append({"src": f"slack:{frac:.0%}", "y_sg": ysg, "y_rt": yrt,
                      "x": greedy_slots(book, book["r"] * book["q"], penalty=fuel_draft_penalty(ysg, yrt),
                                        fill_frac=frac)})

    # --- 候选 2：真 LLM 联合方案 ---
    prompt = build_prompt(book)
    models = [] if "--no-llm" in sys.argv else ["qwen2.5:7b", "mistral:7b"]
    for model in models:
        for t in [0.4, 1.0]:
            seed = int(zlib.crc32(f"{model}|{t}".encode()) % 1_000_000)
            pr = parse(call_ollama(model, prompt, t, seed), book)
            if pr is None:
                print(f"  llm {model}@{t}: 解析失败")
                continue
            x, ysg, yrt = pr
            if ysg + yrt < B_TOTAL - 1e-6:
                ysg = min(TANK["Singapore"], ysg + (B_TOTAL - ysg - yrt))
            plans.append({"src": f"llm:{model}@{t}", "x": x, "y_sg": ysg, "y_rt": yrt})

    rows = []
    for p in plans:
        rev = float(book["r"] * book["q"] @ p["x"])
        cost = VLSFO["Singapore"] * p["y_sg"] + VLSFO["Rotterdam"] * p["y_rt"]
        obj = rev - cost
        gap = (opt - obj) / abs(opt)
        risk, bump, short, p95 = risk_of(book, p["x"], p["y_sg"], p["y_rt"])
        cap_load = (book["w"] * (book["q"] * p["x"])[:, None]).sum(0)
        rows.append({"plan": p["src"], "n_acc": int(p["x"].sum()),
                     "teu": float(book["q"] @ p["x"]), "rev": rev,
                     "y_sg": p["y_sg"], "y_rt": p["y_rt"],
                     "obj": obj, "gap": gap, "risk": risk, "risk_p95": p95,
                     "bump_frac": bump, "fuel_short": short,
                     "max_leg_util": float(np.max(cap_load / CAP_TEU)),
                     "over_cap_base": bool(np.any(cap_load > CAP_TEU + 1e-9))})
    C = pd.DataFrame(rows).sort_values("gap").reset_index(drop=True)
    C.to_csv(os.path.join(OUT, "real_case_plans.csv"), index=False)
    print("\n=== 案例：候选方案（1 条航线、1 个航次）===")
    print(C.round(3).to_string(index=False))

    ig, ir = C.gap.idxmin(), C.risk.idxmin()
    print(f"\ntop-1 冲突: {'YES' if ig != ir else 'NO'}")
    print(f"  min-gap  : {C.loc[ig,'plan']}  gap={C.loc[ig,'gap']:.4f}  risk={C.loc[ig,'risk']:.4f}")
    print(f"  min-risk : {C.loc[ir,'plan']}  gap={C.loc[ir,'gap']:.4f}  risk={C.loc[ir,'risk']:.4f}")
    print(f"  rho(gap,risk) = {spearmanr(C.gap, C.risk).statistic:.3f}")

    json.dump({"B_total_t": B_TOTAL, "b_sg_leg_t": B_SG_LEG, "opt_obj": opt,
               "opt_plan": {"teu": float(book["q"] @ xopt), "y_sg": ysg0, "y_rt": yrt0},
               "leg_nm": LEG_NM.tolist(), "cap_teu": CAP_TEU.tolist(),
               "vlsfo": VLSFO, "port_fee": PORT_FEE.tolist(),
               "n_candidates": len(C),
               "rho_gap_risk": spearmanr(C.gap, C.risk).statistic,
               "conflict": bool(ig != ir),
               "min_gap": C.loc[ig].to_dict(), "min_risk": C.loc[ir].to_dict()},
              open(os.path.join(OUT, "real_case_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.5))
        style = {"llm": ("#d35400", "X", 62), "slack": ("#7f8c8d", "D", 46), "heur": ("#2c7fb8", "o", 42)}
        for _, r in C.iterrows():
            key = "llm" if r.plan.startswith("llm") else ("slack" if r.plan.startswith("slack") else "heur")
            c_, m_, s_ = style[key]
            ax[0].scatter(r.gap, r.risk, c=c_, marker=m_, s=s_, alpha=0.9, zorder=3)
        ax[0].scatter([C.loc[ig, "gap"]], [C.loc[ig, "risk"]], s=210, facecolors="none",
                      edgecolors="k", lw=1.8, zorder=4)
        ax[0].scatter([C.loc[ir, "gap"]], [C.loc[ir, "risk"]], s=210, marker="s", facecolors="none",
                      edgecolors="k", lw=1.8, zorder=4)
        ax[0].annotate("gap's top pick\n(fills the ship)", xy=(C.loc[ig, "gap"], C.loc[ig, "risk"]),
                       xytext=(0.10, 0.235), fontsize=8,
                       arrowprops=dict(arrowstyle="->", lw=0.8))
        ax[0].annotate("lowest risk\n(carries almost nothing)", xy=(C.loc[ir, "gap"], C.loc[ir, "risk"]),
                       xytext=(0.32, 0.055), fontsize=8,
                       arrowprops=dict(arrowstyle="->", lw=0.8))
        ax[0].set_xlabel("optimality gap of the plan")
        ax[0].set_ylabel("realised overrun of the committed contribution")
        ax[0].set_title(f"One voyage, {len(C)} candidate plans", fontsize=10)
        ax[0].set_xlim(-0.03, 0.83)
        ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color="#2c7fb8", label="heuristic"),
                              Line2D([], [], marker="X", ls="", color="#d35400", label="LLM"),
                              Line2D([], [], marker="D", ls="", color="#7f8c8d", label="deliberate slack"),
                              Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=9, label="min gap"),
                              Line2D([], [], marker="s", ls="", mfc="none", mec="k", ms=9, label="min risk")],
                     fontsize=7.5, frameon=False, loc="upper right")

        ax2 = ax[1].twinx()
        ax[1].scatter(C.teu, C.gap, c="#2c7fb8", marker="o", s=40, label="gap (left axis)")
        ax2.scatter(C.teu, C.risk, c="#c0392b", marker="^", s=40, label="realised risk (right axis)")
        ax[1].set_xlabel("committed TEU on the voyage")
        ax[1].set_ylabel("optimality gap", color="#2c7fb8")
        ax2.set_ylabel("realised risk", color="#c0392b")
        ax2.spines["top"].set_visible(False)
        ax[1].set_title("Opposite monotonicity in the commitment lever", fontsize=10)
        ax[1].tick_params(axis="y", colors="#2c7fb8")
        ax2.tick_params(axis="y", colors="#c0392b")
        h1, l1 = ax[1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax[1].legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper center")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_case.png"), dpi=190)
        print("saved fig_case.png")
    except Exception as e:
        print("plot skipped:", e)
