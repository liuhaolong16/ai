# -*- coding: utf-8 -*-
"""
find_failure_source.py — 定位跨温失败的真实失效源

方法：留一特征消融（leave-one-feature-out）
  对 F1 的每个特征 f，移除 f 后重测三档跨温 R²
  判据（预注册）：
    ★ 若移除某特征 f 后，三档跨温 R² 的【最小值】显著回升（> -1），
      则 f 即主要失效源
    ★ 若没有任何单特征移除能做到，则失效源是【多特征联合】，不是单点
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
OUT = os.path.join(_RESULTS, "failure_source.md")
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


def cross(rows, feats):
    import xgboost as xgb
    res = {}
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
        ss = ((yte - yte.mean()) ** 2).sum()
        res[int(t)] = float(1 - ((yte - p) ** 2).sum() / ss) if ss > 0 else float("nan")
    return res


def main():
    rows = load(F1)
    L = ["# 跨温失败的失效源定位（2026-10-11）\n"]
    L.append("方法：留一特征消融。对 F1 每个特征，移除后重测三档跨温 R²。\n")
    L.append("**预注册判据**：若移除某特征后三档跨温 R² 的**最小值**回升到 > −1，则该特征即主要失效源；")
    L.append("若没有任何单特征移除能做到，则失效源是**多特征联合**而非单点。\n")

    base = cross(rows, F1)
    print(f"baseline F1: {base}")
    L.append("## 基线\n")
    L.append("| 特征集 | 留4℃ | 留24℃ | 留43℃ | 最小值 |")
    L.append("|---|---|---|---|---|")
    L.append(f"| F1 全 11 特征 | {base.get(4,float('nan')):.3f} | {base.get(24,float('nan')):.3f} | "
             f"{base.get(43,float('nan')):.3f} | {min(base.values()):.3f} |")

    L.append("\n## 留一特征消融（移除单个特征后）\n")
    L.append("| 移除的特征 | 留4℃ | 留24℃ | 留43℃ | 最小值 | Δ最小值 |")
    L.append("|---|---|---|---|---|---|")
    bmin = min(base.values())
    rowsout = []
    print("\n留一特征消融：")
    for f in F1:
        feats = [x for x in F1 if x != f]
        r = cross(rows, feats)
        mn = min(r.values())
        rowsout.append((f, r, mn))
        L.append(f"| `{f}` | {r.get(4,float('nan')):.3f} | {r.get(24,float('nan')):.3f} | "
                 f"{r.get(43,float('nan')):.3f} | {mn:.3f} | {mn-bmin:+.3f} |")
        print(f"   移除 {f:14s}  min={mn:8.3f}  (Δ{mn-bmin:+.3f})")

    best = max(rowsout, key=lambda x: x[2])
    L.append("\n## 判据结论\n")
    if best[2] > -1.0:
        L.append(f"⇒ ★ **触发**：移除 `{best[0]}` 后最小值回升到 {best[2]:.3f}（> −1），")
        L.append(f"   故 **`{best[0]}` 是主要失效源**。")
        print(f"\n★ 触发：{best[0]} 是主要失效源（min R2={best[2]:.3f}）")
    else:
        L.append(f"⇒ ★ **未触发**：最好的单特征移除（`{best[0]}`）只把最小值抬到 {best[2]:.3f}，仍 < −1。")
        L.append("   ⇒ **失效源是多特征联合，不是单点。**")
        L.append("   这本身是结论：跨温失败不是某一个「坏特征」造成的，")
        L.append("   而是【整套间接特征在温度上的联合分布偏移】。")
        print(f"\n★ 未触发：最好 {best[0]} 仅到 {best[2]:.3f} ⇒ 多特征联合")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
