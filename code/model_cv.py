# -*- coding: utf-8 -*-
"""
model_cv.py — 同温 / 跨温 双协议交叉验证

数据：filtered_v2.csv（MAD 主群筛后），排除 B0050（SOH 上升，见 08 §五）
模型：XGBoost 回归（轻量、可解释树模型，符合论文定位）
两协议：
  A 同温留一电池（within-temperature, leave-one-cell-out）
  B 跨温留一温度（cross-temperature, leave-one-temperature-out）
     —— 训练用两档温度，测试用剩下那一档
特征集三档（做消融）：
  F1 纯间接物理特征（不含 capacity）
  F2 F1 + 循环序号 cycle_idx
  F3 F2 + capacity（★ 注意 capacity 是 SOH 的定义成分，此项会构成近乎同义反复）
输出：results_cv.md + 控制台表
"""
import csv, json, math, os
import numpy as np

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
HERE = _REPO
SRC = os.path.join(_DATA, "filtered_v2.csv")
OUT = os.path.join(_RESULTS, "results_cv.md")
EXCLUDE_CELLS = {"B0050"}

F1 = ["duration_s", "mean_abs_i", "rest_frac", "v_mean", "v_std", "v_min", "v_max",
      "t_mean", "t_max", "t_rise", "n_points"]
F2 = F1 + ["cycle_idx"]
F3 = F2 + ["cap_ah"]
SETS = {"F1 间接物理": F1, "F2 加循环序": F2, "F3 加容量": F3}


def load():
    rows = []
    with open(SRC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cell"] in EXCLUDE_CELLS:
                continue
            try:
                d = {k: float(r[k]) for k in r if k not in ("cell",)}
            except Exception:
                continue
            if not all(math.isfinite(d.get(k, float("nan"))) for k in F3):
                continue
            d["cell"] = r["cell"]
            rows.append(d)
    return rows


def fit_predict(Xtr, ytr, Xte):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                         subsample=0.9, colsample_bytree=0.9, random_state=0,
                         n_jobs=4, verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte), m


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    ss = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")
    return rmse, r2


def main():
    rows = load()
    cells = sorted({r["cell"] for r in rows})
    temps = sorted({r["temp_c"] for r in rows})
    print(f"样本 {len(rows)} 行 · 电池 {len(cells)} · 温度 {temps}")
    line = []
    line.append("# 同温 / 跨温 双协议交叉验证结果\n")
    line.append(f"- 数据：`filtered_v2.csv`（MAD 主群筛后），**排除 {sorted(EXCLUDE_CELLS)}**")
    line.append(f"- 样本 {len(rows)} 行 · 电池 {len(cells)} 只 · 温度 {[int(t) for t in temps]}")
    line.append("- 模型：XGBoost（n=300, depth=4, lr=0.08）\n")

    for name, feats in SETS.items():
        print(f"\n===== {name}（{len(feats)} 特征） =====")
        line.append(f"\n## {name}（{len(feats)} 特征）\n")
        # A 同温留一电池
        line.append("### A 同温留一电池（within-temperature leave-one-cell-out）\n")
        line.append("| 温度 | 电池数 | RMSE | R² |")
        line.append("|---|---|---|---|")
        print("  A 同温留一电池")
        for t in temps:
            sub = [r for r in rows if r["temp_c"] == t]
            cs = sorted({r["cell"] for r in sub})
            if len(cs) < 2:
                continue
            ys, ps = [], []
            for c in cs:
                tr = [r for r in sub if r["cell"] != c]
                te = [r for r in sub if r["cell"] == c]
                if len(tr) < 10 or not te:
                    continue
                Xtr = np.array([[r[k] for k in feats] for r in tr], float)
                ytr = np.array([r["soh"] for r in tr], float)
                Xte = np.array([[r[k] for k in feats] for r in te], float)
                p, _ = fit_predict(Xtr, ytr, Xte)
                ys += [r["soh"] for r in te]; ps += list(p)
            if ys:
                rm, r2 = metrics(ys, ps)
                print(f"    {t:5.0f}C  n={len(cs):2d}  RMSE={rm:.4f}  R2={r2:.3f}")
                line.append(f"| {int(t)} ℃ | {len(cs)} | {rm:.4f} | {r2:.3f} |")

        # B 跨温留一温度
        line.append("\n### B 跨温留一温度（cross-temperature leave-one-temperature-out）\n")
        line.append("| 留出温度 | 训练温度 | RMSE | R² |")
        line.append("|---|---|---|---|")
        print("  B 跨温留一温度")
        for t in temps:
            tr = [r for r in rows if r["temp_c"] != t]
            te = [r for r in rows if r["temp_c"] == t]
            if len(tr) < 20 or len(te) < 5:
                continue
            Xtr = np.array([[r[k] for k in feats] for r in tr], float)
            ytr = np.array([r["soh"] for r in tr], float)
            Xte = np.array([[r[k] for k in feats] for r in te], float)
            yte = np.array([r["soh"] for r in te], float)
            p, _ = fit_predict(Xtr, ytr, Xte)
            rm, r2 = metrics(yte, p)
            trs = "/".join(str(int(x)) for x in temps if x != t)
            print(f"    留{int(t):3.0f}C  RMSE={rm:.4f}  R2={r2:7.3f}")
            line.append(f"| {int(t)} ℃ | {trs} ℃ | {rm:.4f} | {r2:.3f} |")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(line) + "\n")
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
