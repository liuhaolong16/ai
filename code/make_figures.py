# -*- coding: utf-8 -*-
"""
make_figures.py — 用 figkit 重做 4 张图（矢量 PDF，符合期刊规格）

★ 背景：原 4 张 PNG 生成于 2026-10-10 17:40，而数据重建在其后（19:43 起），
  且 fig2 名为 cross_temp 时并无跨温数据、fig3 名为 shap 时并未跑过 SHAP。
  ⇒ 4 张全部重做，改为矢量 PDF（figkit 规格：色盲友好 / 字体≥7pt / 可脚本复现）。

图清单：
  fig1_soh_curves      各电池 SOH-循环曲线，按温度着色
  fig2_dual_protocol   同温 vs 跨温 R²（双协议对比条形）
  fig3_shap            SHAP 全局归因（F1 / F3 两栏）
  fig4_ablation        留一特征消融对跨温 R² 最小值的影响
"""
import csv, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
HERE = _REPO
FIG = _FIGURES
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local figkit.py (vendored)
import figkit

SRC = os.path.join(_DATA, "filtered_v2.csv")
EXCLUDE = {"B0050"}


def load():
    rows = []
    with open(SRC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cell"] in EXCLUDE:
                continue
            try:
                d = {k: float(r[k]) for k in r if k != "cell"}
            except Exception:
                continue
            d["cell"] = r["cell"]
            rows.append(d)
    return rows


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load()
    temps = sorted({r["temp_c"] for r in rows})
    figkit.apply_style()
    made = []

    # ── fig1 SOH 曲线 ──
    fig, ax = figkit.new_fig(width="double", ratio=0.42)
    for ti, t in enumerate(temps):
        sub = [r for r in rows if r["temp_c"] == t]
        cells = sorted({r["cell"] for r in sub})
        for c in cells:
            cc = sorted([r for r in sub if r["cell"] == c], key=lambda x: x["cycle_idx"])
            ax.plot([r["cycle_idx"] for r in cc], [r["soh"] for r in cc],
                    **figkit.line_kw(ti, label=(f"{int(t)} °C" if c == cells[0] else None)))
    p1 = os.path.join(FIG, "fig1_soh_curves.pdf")
    figkit.finish(fig, p1, xlabel="Cycle index", ylabel="SOH", legend=True)
    made.append(p1)

    # ── fig2 双协议对比 ──
    same = {4.0: 0.398, 24.0: 0.874, 43.0: 0.527}      # F1 同温留一电池
    cross = {4.0: -4.459, 24.0: -4.133, 43.0: -8.002}   # F1 跨温留一温度
    fig, ax = figkit.new_fig(width="single", ratio=0.85)
    x = np.arange(len(temps)); w = 0.36
    ax.bar(x - w / 2, [same[t] for t in temps], w, label="Within-temp. (LOCO)",
           color=figkit.PALETTE["blue"])
    ax.bar(x + w / 2, [cross[t] for t in temps], w, label="Cross-temp. (LOTO)",
           color=figkit.PALETTE["vermillion"])
    ax.set_xticks(x); ax.set_xticklabels([f"{int(t)} °C" for t in temps])
    ax.axhline(0, color="0.3", lw=0.75)
    p2 = os.path.join(FIG, "fig2_dual_protocol.pdf")
    figkit.finish(fig, p2, xlabel="Held-out temperature", ylabel="R²", legend=True)
    made.append(p2)

    # ── fig3 SHAP ──
    shap_f1 = [("t_mean", 0.08603), ("duration_s", 0.03577), ("mean_abs_i", 0.03100),
               ("v_std", 0.01879), ("rest_frac", 0.01639), ("n_points", 0.01307),
               ("t_max", 0.01034), ("t_rise", 0.00525), ("v_mean", 0.00341),
               ("v_max", 0.00127), ("v_min", 0.00106)]
    shap_f3 = [("cap_ah", 0.11516), ("t_mean", 0.00907), ("cycle_idx", 0.00299),
               ("v_std", 0.00245), ("rest_frac", 0.00228), ("duration_s", 0.00152)]
    fig, axes = figkit.new_fig(width="double", ratio=0.42, nrows=1, ncols=2)
    for axi, (sub, nm) in enumerate([(shap_f1, "F1 (no capacity)"), (shap_f3, "F3 (with capacity)")]):
        sub = list(reversed(sub))
        axes[axi].barh([s[0] for s in sub], [s[1] for s in sub],
                       color=figkit.PALETTE["blue"] if axi == 0 else figkit.PALETTE["vermillion"],
                       height=0.62)
        axes[axi].set_title(nm, fontsize=8)
    p3 = os.path.join(FIG, "fig3_shap.pdf")
    figkit.finish(fig, p3, xlabel="mean |SHAP| (Ah)")
    made.append(p3)

    # ── fig4 留一消融 ──
    abl = [("duration_s", -5.838), ("rest_frac", -5.807), ("v_mean", -7.340),
           ("v_std", -7.760), ("n_points", -7.833), ("v_min", -7.847),
           ("v_max", -7.906), ("t_rise", -8.410), ("t_max", -9.194),
           ("mean_abs_i", -11.174), ("t_mean", -28.084)]
    abl = sorted(abl, key=lambda x: x[1])
    fig, ax = figkit.new_fig(width="double", ratio=0.38)
    ax.barh([a[0] for a in abl], [a[1] for a in abl],
            color=[figkit.PALETTE["green"] if v > -6 else figkit.PALETTE["grey"]
                   if v > -9 else figkit.PALETTE["vermillion"] for _, v in abl],
            height=0.62)
    ax.axvline(-8.002, color="0.2", lw=0.9, ls="--")
    ax.text(-8.002, len(abl) - 0.4, " baseline −8.002", fontsize=6.5, color="0.2")
    ax.axvline(-1.0, color=figkit.PALETTE["vermillion"], lw=0.9, ls=":")
    ax.text(-1.0, -0.6, "criterion −1", fontsize=6.5, color=figkit.PALETTE["vermillion"])
    p4 = os.path.join(FIG, "fig4_ablation.pdf")
    figkit.finish(fig, p4, xlabel="min cross-temp. R² after removing the feature")
    made.append(p4)

    print("生成：")
    for p in made:
        try:
            figkit.check_pdf_or_raise(p)
            print("  OK  " + os.path.basename(p) + "  " + str(os.path.getsize(p)) + " B  (spec PASS)")
        except Exception as e:
            print("  [WARN]  " + os.path.basename(p) + "  " + str(e)[:120])


if __name__ == "__main__":
    main()
