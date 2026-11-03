#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把海运舱位分配（C 族）的候选换成真 LLM 生成，检验"gap 与风险脱钩"是否仍成立。
本地 ollama，两个模型 × 三个温度，每个组合采样一次；去重后统计。
仅保留名义可行的候选（与 A/B 语料一致）。
"""
import json, os, sys, time, urllib.request
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(OUT, "sea_slot_pilot.py"), encoding="utf-8").read().split("inst_rows, cand_rows")[0]
src = src.replace('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = "."')
exec(src)

MODELS = ["qwen2.5:7b", "mistral:7b"]
TEMPS = [0.4, 0.7, 1.0]
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SMOKE = "--smoke" in sys.argv
CSV = os.path.join(OUT, "llm_slot_candidates.csv")


def build_prompt(inst):
    rows = [f'{i}: legs {int(inst["o"][i])}..{int(inst["d"][i])-1}, qty={inst["q"][i]:.0f} TEU, rate={inst["r"][i]:.0f} USD/TEU'
            for i in range(inst["n"])]
    return ("You are a container-slot allocator. The service has 4 legs (0,1,2,3); each leg has "
            f"capacity {inst['caps'][0]:.0f} TEU. Booking requests below give the occupied legs, "
            "the quantity and the freight rate.\n\n" + "\n".join(rows) +
            "\n\nSelect bookings to maximise total revenue subject to: for every leg, the accepted "
            "quantity does not exceed capacity.\n"
            'Output ONLY JSON: {"accept": [id, ...]}')


def call_ollama(model, prompt, temp, seed, timeout=180):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": temp, "seed": seed, "num_predict": 160}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())["response"]
    except Exception as e:
        return None


def parse(text, n):
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    ids = obj.get("accept")
    if not isinstance(ids, list):
        return None
    x = np.zeros(n)
    for i in ids:
        try:
            j = int(i)
        except Exception:
            continue
        if 0 <= j < n:
            x[j] = 1
    return x


rows = []
for s in range(N_INST):
    inst = gen_slot(1000 + s, n_book=int(18 + 6 * (s % 3)), cap=8000, tight=[1.05, 1.15, 1.30][s % 3])
    opt, _ = solve(inst)
    if not np.isfinite(opt):
        continue
    prompt = build_prompt(inst)
    seen, cands = set(), []
    for model in (MODELS[:1] if SMOKE else MODELS):
        for t in (TEMPS[:1] if SMOKE else TEMPS):
            seed = int(zlib.crc32(f"{s}|{model}|{t}".encode()) % 1_000_000)
            t0 = time.time()
            txt = call_ollama(model, prompt, t, seed)
            x = parse(txt, inst["n"])
            if x is None:
                continue
            key = tuple(x)
            if key in seen:
                continue
            seen.add(key)
            if not np.all(inst["A"] @ x <= 1 + 1e-9):
                rows.append({"inst": 1000 + s, "model": model, "temp": t, "feasible": False,
                             "latency": time.time() - t0})
                continue
            objv = float(inst["r"] @ x)
            g = (opt - objv) / abs(opt)
            ov, av, hv = viol_rate(inst, x)
            cands.append({"inst": 1000 + s, "model": model, "temp": t, "feasible": True,
                          "obj": objv, "gap": g, "overflow": ov, "abs_overflow": av,
                          # 承诺量杠杆：已承诺箱量（TEU）与该方案的最高航段利用率
                          "committed_teu": float(inst["q"] @ x),
                          "util": float(np.max(inst["A"] @ x)),
                          "n_acc": int(x.sum()),
                          "legs_loaded": int(np.sum(inst["A"] @ x > 1e-9)),
                          "latency": time.time() - t0})
            rows.append(cands[-1])
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"[{1000+s}] opt={opt:.0f} llm_cands={len(cands)} "
          f"gaps={[round(c['gap'],3) for c in cands]}", flush=True)
    if SMOKE:
        break

df = pd.DataFrame(rows)
df.to_csv(CSV, index=False)
ok = df[df.feasible == True] if "feasible" in df else df
print(f"\n调用 {len(df)} 次，其中名义可行 {len(ok)}，不可行 {len(df)-len(ok)}")
if len(ok) == 0:
    raise SystemExit("no feasible llm candidate")

inst_rows = []
for inst_id, g in ok.groupby("inst"):
    if len(g) < 2:
        continue
    r1 = spearmanr(g.gap, g.overflow).statistic
    r2 = spearmanr(g.gap, g.abs_overflow).statistic if np.ptp(g.abs_overflow) > 0 else np.nan
    ig, iv = int(np.argmin(g.gap.values)), int(np.argmin(g.overflow.values))
    inst_rows.append({"inst": inst_id, "n": len(g), "rho_ratio": r1, "rho_abs": r2,
                      "flip": ig != iv, "ovr_min_gap": g.overflow.values[ig],
                      "ovr_min_ovr": g.overflow.values[iv],
                      "gap_min_gap": g.gap.values[ig], "gap_min_ovr": g.gap.values[iv]})
I = pd.DataFrame(inst_rows)
I.to_csv(os.path.join(OUT, "llm_slot_instances.csv"), index=False)
if len(I):
    print(f"可统计实例 {len(I)}；rho(gap,甩柜率) 中位 {I.rho_ratio.median():.3f}；"
          f"rho(gap,绝对) {I.rho_abs.median():.3f}；top-1 反转 {I.flip.mean():.3f}")
    print(f"min-gap 甩柜率 {I.ovr_min_gap.median():.4f} vs 最稳 {I.ovr_min_ovr.median():.4f}")
    print(f"min-gap 的 gap {I.gap_min_gap.median():.4f} vs 最稳方案的 gap {I.gap_min_ovr.median():.4f}")
