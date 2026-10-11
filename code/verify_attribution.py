# -*- coding: utf-8 -*-
"""
verify_attribution.py — 验证 09-归因发现.md §五 的两条未核
  ① t_mean 在档内是否是老化的代理 ⇒ 同档内 t_mean 与 cycle_idx 的相关系数
  ② 移除 t_mean 后跨温是否仍失败 ⇒ 干预实验
"""
import csv, math, os
import numpy as np

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
HERE = _REPO
SRC = os.path.join(_DATA, "filtered_v2.csv")
OUT = os.path.join(_RESULTS, "verify_attribution.md")
EXCLUDE = {"B0050"}
F1 = ["duration_s", "mean_abs_i", "rest_frac", "v_mean", "v_std", "v_min", "v_max",
      "t_mean", "t_max", "t_rise", "n_points"]


def load(feats):
    rows = []
    with open(SRC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cell"] in EXCLUDE:
                continue
            try:
                d = {k: float(r[k]) for k in r if k != "cell"}
            except Exception:
                continue
            if not all(math.isfinite(d.get(k, float("nan"))) for k in feats):
                continue
            d["cell"] = r["cell"]
            rows.append(d)
    return rows


def cv_cross(rows, feats):
    import xgboost as xgb
    out = []
    for t in sorted({r["temp_c"] for r in rows}):
        tr = [r for r in rows if r["temp_c"] != t]
        te = [r for r in rows if r["temp_c"] == t]
        if len(tr) < 20 or len(te) < 5:
            continue
        Xtr = np.array([[r[k] for k in feats] for r in tr], float)
        ytr = np.array([r["soh"] for r in tr], float)
        Xte = np.array([[r[k] for k in feats] for r in te], float)
        yte = np.array([r["soh"] for r in te], float)
        m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                             subsample=0.9, colsample_bytree=0.9, random_state=0,
                             n_jobs=4, verbosity=0)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        rmse = float(np.sqrt(((yte - p) ** 2).mean()))
        ss = ((yte - yte.mean()) ** 2).sum()
        r2 = float(1 - ((yte - p) ** 2).sum() / ss) if ss > 0 else float("nan")
        out.append((t, rmse, r2))
    return out


def main():
    rows = load(F1)
    L = ["# 归因发现的验证（2026-10-11）\n"]
    L.append("验证 `09-归因发现.md` §五 的两条未核。\n")

    # ① 档内 t_mean 与 cycle_idx 的相关
    L.append("## ① t_mean 在档内是否是老化的代理\n")
    L.append("同温度档内，`t_mean` 与 `cycle_idx`（循环序号）的 Pearson 相关：\n")
    L.append("| 温度 | n | r(t_mean, cycle_idx) | 判读 |")
    L.append("|---|---|---|---|")
    print("① 档内 t_mean vs cycle_idx 相关")
    for t in sorted({r["temp_c"] for r in rows}):
        sub = [r for r in rows if r["temp_c"] == t]
        a = np.array([r["t_mean"] for r in sub])
        b = np.array([r["cycle_idx"] for r in sub])
        r = float(np.corrcoef(a, b)[0, 1])
        verdict = "支持（档内随老化上升）" if r > 0.30 else ("弱/不支持" if r < 0.15 else "部分支持")
        L.append(f"| {int(t)} ℃ | {len(sub)} | {r:+.4f} | {verdict} |")
        print(f"   {t:5.0f}C  r={r:+.4f}  {verdict}")

    # ② 移除 t_mean / t_max / t_rise 后的跨温
    L.append("\n## ② 移除温度类特征后，跨温是否仍失败\n")
    variants = {
        "F1 原样（含 t_mean/t_max/t_rise）": F1,
        "F1 移除 t_mean": [f for f in F1 if f != "t_mean"],
        "F1 移除全部温度类（t_mean/t_max/t_rise）": [f for f in F1 if f not in ("t_mean", "t_max", "t_rise")],
    }
    L.append("| 特征集 | 留 4 ℃ R² | 留 24 ℃ R² | 留 43 ℃ R² |")
    L.append("|---|---|---|---|")
    for name, feats in variants.items():
        res = cv_cross(rows, feats)
        d = {int(t): r2 for t, _, r2 in res}
        L.append(f"| {name} | {d.get(4, float('nan')):.3f} | {d.get(24, float('nan')):.3f} | {d.get(43, float('nan')):.3f} |")
        print(f"   {name:44s} 4C={d.get(4,float('nan')):7.3f} 24C={d.get(24,float('nan')):7.3f} 43C={d.get(43,float('nan')):7.3f}")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
