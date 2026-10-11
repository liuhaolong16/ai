# -*- coding: utf-8 -*-
"""
filter_v2.py — 按【电池内主群】筛参考容量测试循环（MAD 法）

依据 08-筛选判据修正.md：
  协议一致性是【电池内】的性质，不是跨电池的性质。
  ⇒ 对每只电池，用 (log 时长, log 平均|电流|, 静置占比) 三个特征的
     中位数 ± k·MAD 定义主群，只保留同时落在主群内的循环。

特殊处置：
  B0041 内部两群，前群数量压倒后群 ⇒ 任何主群判据都会选中前群 ⇒ 直接排除整只。

输出 filtered_v2.csv + 逐电池判定表 + 温度分组统计。
"""
import scipy.io as sio
import numpy as np
import glob, os, csv, json

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
BASE = _DATA
OUT = os.path.join(_DATA, "filtered_v2.csv")
MIXED = {"B0038", "B0039", "B0040", "B0042", "B0043", "B0044"}
EXCLUDE = {"B0041"}          # 混杂协议，见 08 §四
K = 3.0                      # MAD 倍数
REST_ABS_I = 0.05


def raw_feats(cyc):
    dd = cyc["data"][0, 0]
    nm = dd.dtype.names or ()
    def g(n):
        return np.asarray(dd[n]).ravel().astype(float) if n in nm else np.array([])
    V, I, T, tm, ca = g("Voltage_measured"), g("Current_measured"), g("Temperature_measured"), g("Time"), g("Capacity")
    if V.size < 10 or ca.size < 1 or tm.size < 2 or I.size < 2 or T.size < 1:
        return None
    cap = float(ca[0])
    if not np.isfinite(cap) or cap <= 0:
        return None
    dur = float(tm[-1] - tm[0])
    if dur <= 0:
        return None
    ai = np.abs(I)
    return dict(cap_ah=cap, duration_s=dur, mean_abs_i=float(ai.mean()),
                rest_frac=float((ai < REST_ABS_I).mean()),
                v_mean=float(V.mean()), v_std=float(V.std()),
                v_min=float(V.min()), v_max=float(V.max()),
                t_mean=float(T.mean()), t_max=float(T.max()),
                t_rise=float(T.max() - T.min()), n_points=int(V.size))


def mad_mask(vals, k=K):
    """中位数 ± k·MAD；MAD 为 0 时退回全体"""
    v = np.asarray(vals, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad <= 1e-12:
        return np.ones(v.size, dtype=bool)
    return np.abs(v - med) <= k * 1.4826 * mad


def main():
    kept, table = [], []
    for f in sorted(glob.glob(os.path.join(BASE, "*.mat"))):
        cell = os.path.basename(f).replace(".mat", "")
        if cell in MIXED or cell in EXCLUDE:
            table.append((cell, None, 0, 0, 0, "SKIP"))
            continue
        d = sio.loadmat(f)
        key = [k for k in d.keys() if not k.startswith("__")][0]
        cyc = d[key][0, 0]["cycle"]
        recs, temps = [], []
        for i in range(cyc.shape[1]):
            c = cyc[0, i]
            if str(c["type"][0]) != "discharge":
                continue
            ft = raw_feats(c)
            if ft is None:
                continue
            try:
                ft["temp_c"] = round(float(c["ambient_temperature"][0, 0]), 1)
                temps.append(ft["temp_c"])
            except Exception:
                ft["temp_c"] = np.nan
            recs.append(ft)
        if not recs:
            table.append((cell, None, 0, 0, 0, "无有效循环"))
            continue
        ld = np.log(np.array([r["duration_s"] for r in recs]))
        li = np.log(np.array([r["mean_abs_i"] for r in recs]))
        rf = np.array([r["rest_frac"] for r in recs])
        m = mad_mask(ld) & mad_mask(li) & mad_mask(rf)
        T = max(set(temps), key=temps.count) if temps else float("nan")
        sub = [r for r, keep in zip(recs, m) if keep]
        # SOH 单调性检查（允许 ≤0.02 的抖动）
        if sub:
            sohs = [r["cap_ah"] / 2.0 for r in sub]
            mono = sohs[-1] <= sohs[0] + 0.02
        else:
            mono = False
        for j, r in enumerate(sub, 1):
            r2 = dict(r); r2["cycle_idx"] = j; r2["cell"] = cell; r2["soh"] = r["cap_ah"] / 2.0
            kept.append(r2)
        table.append((cell, T, len(recs), len(sub), len(recs) - len(sub),
                      ("OK" if mono else "⚠SOH上升")))
    cols = ["cell", "temp_c", "cycle_idx", "cap_ah", "soh", "duration_s", "mean_abs_i",
            "rest_frac", "v_mean", "v_std", "v_min", "v_max", "t_mean", "t_max", "t_rise", "n_points"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in cols})
    print("WROTE", OUT, "rows", len(kept))
    print()
    print(f"{'cell':7s} {'T':>5s} {'原':>5s} {'留':>5s} {'剔':>5s}  判定")
    for cell, T, n0, n1, nd, st in table:
        Ts = f"{T:5.0f}" if isinstance(T, float) else "    -"
        print(f"{cell:7s} {Ts} {n0:5d} {n1:5d} {nd:5d}  {st}")
    print()
    byT = {}
    for r in kept:
        byT.setdefault(r["temp_c"], {}).setdefault(r["cell"], 0)
        byT[r["temp_c"]][r["cell"]] += 1
    print("=== 最终可用（按温度） ===")
    for t in sorted(byT, key=lambda x: (np.isnan(x), x)):
        cells = byT[t]
        print(f"  {t:5.1f} C  电池 {len(cells):2d} 只  循环 {sum(cells.values()):5d}  | " +
              ", ".join(f"{c}:{n}" for c, n in sorted(cells.items())))


if __name__ == "__main__":
    main()
