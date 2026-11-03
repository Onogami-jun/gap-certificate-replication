#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D 族（加油港选择）的真 LLM 候选：本地 ollama 两模型 × 三温度，去重后统计 gap 与超支风险。"""
import json, os, sys, time, urllib.request
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(OUT, "sea_bunker_pilot.py"), encoding="utf-8").read().split("inst_rows, cand_rows")[0]
src = src.replace('OUT = os.path.dirname(os.path.abspath(__file__))', 'OUT = "."')
exec(src)

MODELS = ["qwen2.5:7b", "mistral:7b"]
TEMPS = [0.4, 0.7, 1.0]
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SMOKE = "--smoke" in sys.argv
CSV = os.path.join(OUT, "llm_bunker_candidates.csv")


def build_prompt(inst):
    rows = [f'{PORTS[j][0]}: price={inst["p"][j]:.0f} USD/t, max_lift={inst["cap"][j]:.0f} t, '
            f'port_fee={inst["fee"][j]:.0f} USD, deviation={inst["dev"][j]:.0f} nm' for j in range(M)]
    return ("You are a bunker procurement planner for a container service. The vessel needs "
            f"{inst['B']:.0f} tonnes of VLSFO this voyage. Candidate bunkering ports:\n\n"
            + "\n".join(rows) +
            "\n\nHard rules: (a) total lift must be at least the required tonnes; "
            "(b) the lift at any port must NOT exceed that port's max_lift; "
            "(c) if a single port cannot supply the whole requirement, split the lift across several ports.\n"
            'Output ONLY JSON: {"bunkers": {"PORT_NAME": tonnes, ...}}')


def call_ollama(model, prompt, temp, seed, timeout=180):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": temp, "seed": seed, "num_predict": 200}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())["response"]
    except Exception:
        return None


def parse(text, inst):
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    d = obj.get("bunkers")
    if not isinstance(d, dict):
        return None
    y = np.zeros(M)
    for name, v in d.items():
        key = str(name).replace(" ", "").lower()
        for j in range(M):
            if PORTS[j][0].lower() == key:
                try:
                    y[j] = max(0.0, float(v))
                except Exception:
                    pass
    z = (y > 1e-6).astype(int)
    return z, y


rows = []
if os.path.exists(CSV) and "--fresh" not in sys.argv:
    rows = pd.read_csv(CSV).to_dict("records")
done = {int(r["inst"]) for r in rows}
for s in range(N_INST):
    if 2000 + s in done:
        continue
    inst = gen(2000 + s)
    opt, _, _ = solve(inst)
    if not np.isfinite(opt):
        continue
    prompt = build_prompt(inst)
    seen, n_ok = set(), 0
    for model in (MODELS[:1] if SMOKE else MODELS):
        for t in (TEMPS[:1] if SMOKE else TEMPS):
            seed = int(zlib.crc32(f"{s}|{model}|{t}".encode()) % 1_000_000)
            t0 = time.time()
            txt = call_ollama(model, prompt, t, seed)
            if SMOKE:
                print("RAW:", (txt or "")[:600])
            pr = parse(txt, inst)
            if pr is None:
                continue
            z, y = pr
            # 物理裁剪：超出该港可加油上限的部分装不进油舱，只按实际装载量执行
            y = np.minimum(y, inst["cap"])
            z = (y > 1e-6).astype(int)
            if y.sum() <= 1e-6:
                continue          # 空方案（未给出任何可执行加油量），不构成候选
            key = (tuple(z), tuple(np.round(y, 1)))
            if key in seen:
                continue
            seen.add(key)
            # 执行口径：船必须开航，名义缺口按紧急现货溢价强制补足并计入执行成本
            short = max(0.0, inst["B"] - y.sum())
            cost = eval_cost(inst, z, y) + short * SPOT_PREMIUM * inst["p"].max()
            g = (cost - opt) / opt
            ov, av, hv = realized(inst, z, y)
            n_ok += 1
            rows.append({"inst": 2000 + s, "model": model, "temp": t, "feasible": short <= 1e-6,
                         "short_ratio": short / inst["B"],
                         "cost": cost, "gap": g, "overrun": ov, "abs_overrun": av,
                         "ports": "+".join(PORTS[j][0] for j in range(M) if z[j] == 1),
                         # 承诺量杠杆：本航次锁定的加油量（吨）与相对需油量的覆盖率
                         "lift_t": float(y.sum()), "fill": float(y.sum() / inst["B"]),
                         "n_ports": int(z.sum()),
                         "latency": time.time() - t0})
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"[{2000+s}] opt={opt:.0f} feasible_llm={n_ok}", flush=True)
    if SMOKE:
        break

df = pd.DataFrame(rows)
df.to_csv(CSV, index=False)
ok = df[df.gap.notna()]  # 执行口径下所有解析成功的计划都可执行，仅缺口需现货补足
print(f"\n调用 {len(df)} 次，可执行 {len(ok)}，名义足量 {int(df.feasible.sum())}，"
      f"平均缺口率 {df.short_ratio.mean():.3f}")
if len(ok) == 0:
    raise SystemExit("no parsable llm candidate")

inst_rows = []
for inst_id, g in ok.groupby("inst"):
    if len(g) < 2:
        continue
    r1 = spearmanr(g.gap, g.overrun).statistic
    r2 = spearmanr(g.gap, g.abs_overrun).statistic if np.ptp(g.abs_overrun) > 0 else np.nan
    ig, iv = int(np.argmin(g.gap.values)), int(np.argmin(g.overrun.values))
    inst_rows.append({"inst": inst_id, "n": len(g), "rho_ratio": r1, "rho_abs": r2,
                      "flip": ig != iv, "ovr_min_gap": g.overrun.values[ig],
                      "ovr_min_ovr": g.overrun.values[iv],
                      "gap_min_gap": g.gap.values[ig], "gap_min_ovr": g.gap.values[iv]})
I = pd.DataFrame(inst_rows)
I.to_csv(os.path.join(OUT, "llm_bunker_instances.csv"), index=False)
if len(I):
    print(f"可统计实例 {len(I)}；rho(gap,超支率) 中位 {I.rho_ratio.median():.3f}；"
          f"rho(gap,绝对) {I.rho_abs.median():.3f}；top-1 反转 {I.flip.mean():.3f}")
