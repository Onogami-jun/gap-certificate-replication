#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""漂移标定：把一个航次的燃油价格通道从"论文假定的 0-10/0-20% 带"
换成 FRED Brent 日度序列的 24 交易日经验分布，看结论是否仍然成立。

为什么只换价格通道：
  * 论文的四个通道（箱量、有效舱位、需油、油价）在代码里各自独立抽样；
  * 其中只有油价有公开、日度、可核验的真实序列（Brent / VLSFO）；
  * 另外三个通道的真实联合分布没有公开数据，硬编一个"经验分布"会是伪标定。
所以本脚本是一个"单通道替换 + 其余通道保持论文假定"的敏感性分析，不冒充全通道标定。

暴露的解析分解（与 real_case.risk_of 完全等价）：
  realised - committed = -1.3*lost + (fuel(p2) - fuel(p0))
  exposure = [1.3*lost + fuel(p2) - fuel0] / |committed_margin|
  其中 lost = bump_frac * revenue，fuel 对油价的依赖是乘子乘以每港实际加油量的线性函数。
因此给定 plan 的 (rev, y_sg, y_rt, bump_frac, fuel_short)，油价通道可精确重算，
无需重建 60 维的接受向量。
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import real_case as rc

OUT = os.path.dirname(os.path.abspath(__file__))
H = 24          # 航次天数：sum(leg_nm)/(18.5*24) ≈ 24.1 天
N_SCEN = 4000
SEED = 20261028

# ---------------- 1. 经验分布 ----------------
raw = pd.read_csv(os.path.join(OUT, "brent_raw.txt"), na_values=["."])
raw = raw.dropna(subset=["DCOILBRENTEU"])
px = raw.DCOILBRENTEU.to_numpy(float)
lp = np.log(px)
r24 = lp[H:] - lp[:-H]
r24 = r24[np.isfinite(r24)]
mu = float(r24.mean())
print(f"Brent 日度: n={px.size}  {raw.observation_date.iloc[0]} -> {raw.observation_date.iloc[-1]}")
print(f"24 交易日对数收益: n={r24.size}  mean={mu:+.4f}  sd={r24.std(ddof=1):.4f}")
qs = np.percentile(r24, [1, 5, 10, 25, 50, 75, 90, 95, 99])
print("  分位 1/5/10/25/50/75/90/95/99 (%): " +
      " ".join(f"{100*np.expm1(q):+.1f}" for q in qs))
share = {t: float(np.mean(np.abs(r24) > np.log1p(t))) for t in (0.10, 0.15, 0.20)}
print("  |r|>10%: {:.1%}  >15%: {:.1%}  >20%: {:.1%}".format(*share.values()))
down = float(np.mean(r24 < 0))
print(f"  下跌窗口占比: {down:.1%}")

# ---------------- 2. 各方案的聚合量 ----------------
P = pd.read_csv(os.path.join(OUT, "real_case_plans.csv"))
fees = float(rc.PORT_FEE.sum())
p0 = {k: float(v) for k, v in rc.VLSFO.items()}
tank = {k: float(v) for k, v in rc.TANK.items()}


def fuel_cost(m_sg, m_rt, y_sg, y_rt, short):
    """实现油价下的燃油支出（含缺口油的现货溢价）。"""
    a = p0["Singapore"] * min(y_sg, tank["Singapore"])
    b = p0["Rotterdam"] * min(y_rt, tank["Rotterdam"])
    spot = max(m_sg * p0["Singapore"], m_rt * p0["Rotterdam"])
    return m_sg * a + m_rt * b + short * rc.SPOT_PREMIUM * spot


def exposure(mults, y_sg, y_rt, short, lost, cm):
    f0 = fuel_cost(1.0, 1.0, y_sg, y_rt, short)
    out = []
    for m_sg, m_rt in mults:
        out.append((1.3 * lost + fuel_cost(m_sg, m_rt, y_sg, y_rt, short) - f0) / abs(cm))
    return np.asarray(out)


rng = np.random.default_rng(SEED)

# 论文假定：价格乘子 1 + dl*U(0,1)，dl 交替取 10% / 20%（与 risk_of 同构）
n_half = N_SCEN // 2
band = np.zeros((N_SCEN, 2))
band[:n_half, 0] = 1 + 0.10 * rng.uniform(0, 1, n_half)
band[:n_half, 1] = 1 + 0.10 * rng.uniform(0, 1, n_half)
band[n_half:, 0] = 1 + 0.20 * rng.uniform(0, 1, n_half)
band[n_half:, 1] = 1 + 0.20 * rng.uniform(0, 1, n_half)

draw = rng.choice(r24, size=N_SCEN, replace=True)
arms = {
    # 经验、原样（含样本期内的上行趋势）
    "emp": np.column_stack([np.exp(draw), np.exp(draw)]),
    # 经验、去均值（只保留波动，不保留样本期趋势）
    "emp_dm": np.column_stack([np.exp(draw - mu), np.exp(draw - mu)]),
    # 经验、只取不利方向（与论文"只上浮"的符号约定对齐）
    "emp_up": np.column_stack([np.exp(np.abs(draw)), np.exp(np.abs(draw))]),
}
# 传递率 0.8 的敏感性：油价只有 80% 传导到加油支出
arms["emp_dm_pt08"] = 1.0 + 0.8 * (arms["emp_dm"] - 1.0)

rows = []
for _, p in P.iterrows():
    rev, ysg, yrt = float(p.rev), float(p.y_sg), float(p.y_rt)
    short = float(p.fuel_short) * rc.B_TOTAL
    lost = float(p.bump_frac) * rev
    cm = rev - (p0["Singapore"] * ysg + p0["Rotterdam"] * yrt) - fees
    d = {"plan": p.plan, "gap": float(p.gap), "risk_paper": float(p.risk),
         "fuel_t": ysg + yrt,
         # 两个通道各自的贡献（同一 |committed margin| 分母）
         "bump_share": 1.3 * lost / abs(cm),
         "price_share": float(exposure(band, ysg, yrt, short, 0.0, cm).mean())}
    for k, m in [("band", band)] + list(arms.items()):
        e = exposure(m, ysg, yrt, short, lost, cm)
        d[f"{k}_mean"] = float(e.mean())
        d[f"{k}_p95"] = float(np.percentile(e, 95))
        d[f"{k}_p99"] = float(np.percentile(e, 99))
    rows.append(d)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "drift_cal.csv"), index=False)

# ---------------- 3. 比较 ----------------
print("\n=== 论文假定 vs 经验漂移（同一批 19 个方案）===")
print(f"{'arm':<14}{'median':>9}{'p95 med':>9}{'p99 med':>9}{'ratio':>8}"
      f"{'rho':>7}{'min-risk same':>15}")
summary = []
for k in ["band"] + list(arms):
    med = float(D[f"{k}_mean"].median())
    p95 = float(D[f"{k}_p95"].median())
    p99 = float(D[f"{k}_p99"].median())
    rho = float(spearmanr(D.band_mean, D[f"{k}_mean"]).statistic)
    ir_b, ir_k = int(D.band_mean.idxmin()), int(D[f"{k}_mean"].idxmin())
    same = D.plan.iloc[ir_b] == D.plan.iloc[ir_k]
    ratio = med / float(D.band_mean.median())
    summary.append(dict(arm=k, median=med, p95=p95, p99=p99, ratio=ratio, rho=rho, same=same))
    print(f"{k:<14}{med:9.4f}{p95:9.4f}{p99:9.4f}{ratio:8.3f}{rho:7.3f}"
          f"{('yes' if same else 'no'):>15}")
S = pd.DataFrame(summary)
S.to_csv(os.path.join(OUT, "drift_cal_summary.csv"), index=False)

print("\n最小风险方案（把价格通道换成经验漂移后是否换人）：")
for k in ["band"] + list(arms):
    i = int(D[f"{k}_mean"].idxmin())
    print(f"  {k:<14}{D.plan.iloc[i]:<24} exposure={D[f'{k}_mean'].iloc[i]:.4f}  "
          f"gap={D.gap.iloc[i]:.4f}  fuel_t={D.fuel_t.iloc[i]:.0f}")

# 机制方向：加油量 → 价格通道暴露的斜率
print("\n价格通道暴露 vs 总加油量: rho = "
      f"{spearmanr(D.fuel_t, D.emp_dm_mean).statistic:+.3f} (band: "
      f"{spearmanr(D.fuel_t, D.band_mean).statistic:+.3f})")

# gap 的选择在各臂下暴露排名（1 = 最安全）
print("\ngap 最小的方案在各臂下的暴露排名（n=19；排名越高越危险）：")
ig = int(D.gap.idxmin())
for k in ["band"] + list(arms):
    rank = int((D[f"{k}_mean"] > D[f"{k}_mean"].iloc[ig]).sum()) + 1
    print(f"  {k:<14}{D.plan.iloc[ig]:<24}rank {rank}/19  exposure={D[f'{k}_mean'].iloc[ig]:.4f}")

# 油价通道在总暴露中的权重（其余三个通道按论文假定固定）
abs_r = np.abs(r24)
print("\n|24 日对数收益|: 中位 {:.1%}  p90 {:.1%}  p95 {:.1%}".format(
    float(np.median(abs_r)), float(np.percentile(abs_r, 90)), float(np.percentile(abs_r, 95))))
print("超过 ±20% 的窗口: {:.1%}；超过 ±10%: {:.1%}".format(
    float(np.mean(abs_r > np.log(1.20))), float(np.mean(abs_r > np.log(1.10)))))

print("两通道在总暴露中的占比（中位）: 舱位 {:.1%}  油价 {:.1%}".format(
    float((D.bump_share / (D.bump_share + D.price_share)).median()),
    float((D.price_share / (D.bump_share + D.price_share)).median())))
