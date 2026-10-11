# -*- coding: utf-8 -*-
"""
shap_analysis.py — 归因分析
主路径：SHAP（TreeExplainer）
降级路径：若 shap 未安装，明确报告并改用 XGBoost gain 重要性（不静默冒充 SHAP）

输出 shap_results.md：三档温度的全局归因 + 跨温时的归因变化
"""
import csv, math, os, sys
import numpy as np

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
HERE = _REPO
SRC = os.path.join(_DATA, "filtered_v2.csv")
OUT = os.path.join(_RESULTS, "shap_results.md")
EXCLUDE_CELLS = {"B0050"}

F1 = ["duration_s", "mean_abs_i", "rest_frac", "v_mean", "v_std", "v_min", "v_max",
      "t_mean", "t_max", "t_rise", "n_points"]
F3 = F1 + ["cycle_idx", "cap_ah"]

try:
    import shap
    HAVE_SHAP = True
except Exception as e:
    HAVE_SHAP = False
    SHAP_ERR = str(e)


def load(feats):
    rows = []
    with open(SRC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cell"] in EXCLUDE_CELLS:
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


def main():
    import xgboost as xgb
    lines = ["# 归因分析结果\n"]
    lines.append(f"- 方法：{'SHAP TreeExplainer' if HAVE_SHAP else 'XGBoost gain（★ SHAP 不可用，降级）'}")
    if not HAVE_SHAP:
        lines.append(f"- ⚠ SHAP 导入失败：`{SHAP_ERR}`")
    lines.append("")

    for setname, feats in [("F1 间接物理（不含容量）", F1), ("F3 加容量", F3)]:
        rows = load(feats)
        temps = sorted({r["temp_c"] for r in rows})
        X = np.array([[r[k] for k in feats] for r in rows], float)
        y = np.array([r["soh"] for r in rows], float)
        m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                             subsample=0.9, colsample_bytree=0.9, random_state=0,
                             n_jobs=4, verbosity=0)
        m.fit(X, y)
        print(f"===== {setname}  n={len(rows)}")
        lines.append(f"\n## {setname}\n")
        lines.append(f"全量拟合 n={len(rows)}\n")

        if HAVE_SHAP:
            ex = shap.TreeExplainer(m)
            sv = ex.shap_values(X)
            imp = np.abs(sv).mean(axis=0)
            lines.append("| 特征 | 平均 |SHAP| |")
            lines.append("|---|---|")
            for i in np.argsort(-imp):
                lines.append(f"| `{feats[i]}` | {imp[i]:.5f} |")
                print(f"   {feats[i]:26s} {imp[i]:.5f}")
            # 按温度分组
            lines.append("\n### 分温度归因（平均 |SHAP|）\n")
            hdr = "| 特征 | " + " | ".join(f"{int(t)} ℃" for t in temps) + " |"
            lines.append(hdr)
            lines.append("|---" * (len(temps) + 1) + "|")
            for i, f in enumerate(feats):
                cells = []
                for t in temps:
                    idx = [j for j, r in enumerate(rows) if r["temp_c"] == t]
                    cells.append(f"{np.abs(sv[idx, i]).mean():.5f}")
                lines.append(f"| `{f}` | " + " | ".join(cells) + " |")
        else:
            imp = m.feature_importances_
            lines.append("| 特征 | gain 重要性 |")
            lines.append("|---|---|")
            for i in np.argsort(-imp):
                lines.append(f"| `{feats[i]}` | {imp[i]:.5f} |")
                print(f"   {feats[i]:26s} {imp[i]:.5f}")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
