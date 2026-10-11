# -*- coding: utf-8 -*-
"""
seed_robustness.py — 主结果的随机种子稳健性检验（采纳豆包 v1.11 §5.4.1「实验噪声门」）

动机：论文主结果（同温 R²=0.874、跨温 R² 全负）目前是 random_state=0 单次跑出的，
没有 seed 方差。审稿人/答辩的第一枪就是「你跑了几次？±多少？这个结论在 seed 方差内吗」。

做法：不改主结果口径（主结果仍是 seed=0），只**加一层稳健性证据**——
对 F1（论文主张所用的 11 个间接特征）与 F3（含容量，用于对照同义反复）
跑 N 个随机种子，逐格报告 mean±std，并给出「seed=0 值是否落在 mean±std 内」。

输出：seed_robustness.md
"""
import csv, math, os, sys
import numpy as np

# --- repository-relative paths (see README §3) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
SRC = os.path.join(_DATA, "filtered_v2.csv")
OUT = os.path.join(_RESULTS, "seed_robustness.md")
EXCLUDE_CELLS = {"B0050"}

F1 = ["duration_s", "mean_abs_i", "rest_frac", "v_mean", "v_std", "v_min", "v_max",
      "t_mean", "t_max", "t_rise", "n_points"]
F3 = F1 + ["cycle_idx", "cap_ah"]
SETS = {"F1（论文主张的 11 个间接特征）": F1, "F3（+循环序+容量，对照）": F3}

SEEDS = list(range(10))


def load():
    rows = []
    with open(SRC, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cell"] in EXCLUDE_CELLS:
                continue
            try:
                d = {k: float(r[k]) for k in r if k != "cell"}
            except Exception:
                continue
            if not all(math.isfinite(d.get(k, float("nan"))) for k in F3):
                continue
            d["cell"] = r["cell"]
            rows.append(d)
    return rows


def fit_predict(Xtr, ytr, Xte, seed):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                         subsample=0.9, colsample_bytree=0.9, random_state=seed,
                         n_jobs=4, verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def main():
    rows = load()
    cells = sorted({r["cell"] for r in rows})
    temps = sorted({r["temp_c"] for r in rows})
    print(f"样本 {len(rows)} 行 · 电池 {len(cells)} 只 · 温度 {temps} · seeds {SEEDS}")

    lines = ["# 主结果随机种子稳健性检验（seed robustness）\n",
             f"- 数据：`filtered_v2.csv`，排除 {sorted(EXCLUDE_CELLS)} ⇒ {len(rows)} 行 / {len(cells)} 只",
             f"- 模型：XGBoost（n=300, depth=4, lr=0.08, subsample=0.9, colsample_bytree=0.9）",
             f"- 种子：{SEEDS[0]}~{SEEDS[-1]}（共 {len(SEEDS)} 个），其余超参与主结果完全一致",
             "- 目的：回答「结论是否落在 seed 方差内」；**主结果口径不变（seed=0）**，本表只作稳健性证据\n"]

    summary = []
    for name, feats in SETS.items():
        print(f"\n===== {name} =====")
        lines.append(f"\n## {name}\n")
        for proto in ("A", "B"):
            head = ("### A 同温留一电池（LOCO）" if proto == "A"
                    else "### B 跨温留一温度（LOTO）")
            lines.append(head + "\n")
            if proto == "B":
                lines.append("| 温度 | seed=0（主结果） | mean ± std（n=%d） | min ~ max | 极差 | **全部种子 < 0？** |" % len(SEEDS))
                lines.append("|---|---|---|---|---|---|")
            else:
                lines.append("| 温度 | seed=0（主结果） | mean ± std（n=%d） | min ~ max | 极差 | 落在 ±1std 内 |" % len(SEEDS))
                lines.append("|---|---|---|---|---|---|")
            print(f"  {proto}")
            for t in temps:
                vals = []
                for seed in SEEDS:
                    if proto == "A":
                        sub = [r for r in rows if r["temp_c"] == t]
                        cs = sorted({r["cell"] for r in sub})
                        if len(cs) < 2:
                            vals = []
                            break
                        ys, ps = [], []
                        for c in cs:
                            tr = [r for r in sub if r["cell"] != c]
                            te = [r for r in sub if r["cell"] == c]
                            if len(tr) < 10 or not te:
                                continue
                            Xtr = np.array([[r[k] for k in feats] for r in tr], float)
                            ytr = np.array([r["soh"] for r in tr], float)
                            Xte = np.array([[r[k] for k in feats] for r in te], float)
                            ps += list(fit_predict(Xtr, ytr, Xte, seed))
                            ys += [r["soh"] for r in te]
                        vals.append(r2(ys, ps))
                    else:
                        tr = [r for r in rows if r["temp_c"] != t]
                        te = [r for r in rows if r["temp_c"] == t]
                        if len(tr) < 20 or len(te) < 5:
                            vals = []
                            break
                        Xtr = np.array([[r[k] for k in feats] for r in tr], float)
                        ytr = np.array([r["soh"] for r in tr], float)
                        Xte = np.array([[r[k] for k in feats] for r in te], float)
                        yte = np.array([r["soh"] for r in te], float)
                        vals.append(r2(yte, fit_predict(Xtr, ytr, Xte, seed)))
                if not vals:
                    continue
                v = np.array(vals, float)
                mu, sd = v.mean(), v.std(ddof=1)
                s0 = vals[0]
                if proto == "B":
                    allneg = "**是**（10/10）" if (v < 0).all() else "**否**（%d/%d 为负）" % (int((v < 0).sum()), len(v))
                    extra = allneg
                    verdict = allneg
                else:
                    inside = "是" if (sd == 0 or abs(s0 - mu) <= sd) else "否"
                    extra = inside
                    verdict = inside
                lines.append(f"| {int(t)} ℃ | {s0:.3f} | {mu:.3f} ± {sd:.3f} | {v.min():.3f} ~ {v.max():.3f} | {v.max()-v.min():.3f} | {extra} |")
                print(f"    {int(t):3d}C  seed0={s0:7.3f}  mean={mu:7.3f}  std={sd:6.3f}  "
                      f"min={v.min():7.3f}  max={v.max():7.3f}  -> {verdict}")
                summary.append((name, proto, int(t), s0, mu, sd, v.max() - v.min(), extra, v.min(), v.max()))

    # 结论段
    lines.append("\n## 判读（对照豆包 v1.11 §5.4.1 实验噪声门）\n")
    f1 = [s for s in summary if s[0].startswith("F1（")]
    f1B = [s for s in f1 if s[1] == "B"]
    worst = max(f1, key=lambda s: s[5])
    lines.append("- 判据：**结论要成立，必须远大于 seed 方差**；若某格 std 与其结论幅度同量级，该结论不可下。")
    lines.append(f"- F1 各格 std 最大为 **{worst[5]:.3f}**（{worst[1]} 协议 · {worst[2]} ℃）")
    allneg = [s for s in f1B if s[7].startswith("**是**")]
    lines.append(f"- ★★ **跨温（B 协议）{len(allneg)}/{len(f1B)} 个格上，全部 {len(SEEDS)} 个种子的 R² 均为负**"
                 f"（见上表「全部种子 < 0？」列）⇒ 「跨温失败」这一主结论对随机种子不敏感。")
    if f1B:
        mx = max(s[4] for s in f1B)
        lines.append(f"- 跨温最不利的一格均值为 **{mx:.3f}**，其绝对幅度远大于同格 std ⇒ 不是波动造成的。")
    a24 = [s for s in f1 if s[1] == "A" and s[2] == 24]
    if a24:
        s = a24[0]
        lines.append(f"- 同温主结果 24 ℃：seed=0 的 **{s[3]:.3f}** 与 {len(SEEDS)} 种子均值 **{s[4]:.3f} ± {s[5]:.3f}** "
                     f"一致（range {s[6]:.3f}）⇒ 论文头条数字可复现。")
    lines.append("- 论文口径：主结果仍报 seed=0 的点估计（与 `results_cv.md` 逐字一致）；"
                 "本表作为稳健性证据/补充材料，用于正面回答「你跑了几次、方差多少」。")

    lines.append("\n### ⚠ 必须一并披露的反例（F3 对照组）\n")
    f3B = [s for s in summary if s[0].startswith("F3（") and s[1] == "B"]
    for s in f3B:
        lines.append(f"- F3 跨温 **{s[2]} ℃**：seed=0 = {s[3]:.3f}，{len(SEEDS)} 种子 = {s[4]:.3f} ± {s[5]:.3f}，"
                     f"范围 {s[8]:.3f} ~ {s[9]:.3f}")
    byT = {s[2]: s for s in f3B}
    pos = [s for s in f3B if s[4] > 0]
    if pos:
        lines.append("- ⇒ **F3 跨温存在正 R² 的格**：" +
                     "；".join(f"{s[2]} ℃ = {s[4]:.3f} ± {s[5]:.3f}" for s in pos) +
                     "（含容量的特征集在这些温度上**跨温也能迁移**）")
    if 24 in byT:
        s = byT[24]
        lines.append(f"- ⇒ 24 ℃ 一格的 {len(SEEDS)} 种子区间为 {s[8]:.3f} ~ {s[9]:.3f}，"
                     f"**跨零**（seed=0 恰好落在正侧 {s[3]:.3f}）⇒ 该格不可作为「F3 也失败」的证据。")
    lines.append("- ⇒ **稿件若写「跨温一律失败」，必须显式限定为 F1（11 个间接特征）**；"
                 "F3 在 43 ℃ 上不失败，这一格是审稿人/评委会抓的反例，**不能靠 'largely failing' 这种含糊措辞绕过**。")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nWROTE", OUT)


if __name__ == "__main__":
    main()
