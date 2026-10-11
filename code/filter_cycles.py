# -*- coding: utf-8 -*-
"""
filter_cycles.py — 按【放电协议】筛出「参考容量测试」循环

依据（2026-10-10 实测，见 07-数据质量核实.md 的续篇）：
  B0041 的前 ~55 个 discharge 是 4 A【脉冲】放电（时长 ~1000 s，深放到 1.95 V），
  放出 0.05 Ah —— 那是脉冲特性测试，不是容量测试；
  后 ~12 个是 1 A【连续】放电（时长 ~5000 s），放出 0.85 Ah —— 那才是容量测试。

⇒ 因此不能用「容量百分位」这种结果型阈值（它会把不同协议混在一起筛），
   要用【协议特征】筛：放电时长、电流量级、以及是否存在长时间静置（脉冲特征）。

判据（三条全过才保留）：
  C1 时长 >= 1200 s                    —— 排除极短循环
  C2 平均 |电流| <= 2.5 A              —— 排除大倍率脉冲
  C3 「静置占比」<= 0.35               —— 脉冲测试里大量时间是静置（I≈0）
                                         参考容量测试电流基本连续

输出 filtered.csv + 逐电池判定表（打印）
"""
import scipy.io as sio
import numpy as np
import glob, os, csv

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
BASE = _DATA
OUT = os.path.join(_DATA, "filtered.csv")
NOMINAL = 2.0
MIXED = {"B0038", "B0039", "B0040", "B0042", "B0043", "B0044"}

C1_MIN_DUR = 1200.0
C2_MAX_ABS_I = 2.5
C3_MAX_REST_FRAC = 0.35
REST_ABS_I = 0.05          # |I| 低于此视为静置


def protocol_feats(cyc):
    dd = cyc["data"][0, 0]
    names = dd.dtype.names or ()
    def g(n):
        return np.asarray(dd[n]).ravel().astype(float) if n in names else np.array([])
    V = g("Voltage_measured")
    I = g("Current_measured")
    T = g("Temperature_measured")
    tm = g("Time")
    caparr = g("Capacity")
    if V.size < 10 or caparr.size < 1:
        return None
    cap = float(caparr[0])
    if not np.isfinite(cap) or cap <= 0:
        return None
    if tm.size < 2 or I.size < 2 or T.size < 1:
        return None
    dur = float(tm[-1] - tm[0])
    if dur <= 0:
        return None
    absI = np.abs(I)
    mean_abs_i = float(absI.mean())
    rest_frac = float((absI < REST_ABS_I).mean())
    return dict(
        cap_ah=cap, soh=cap / NOMINAL, duration_s=dur,
        mean_abs_i=mean_abs_i, rest_frac=rest_frac,
        v_mean=float(V.mean()), v_std=float(V.std()),
        v_min=float(V.min()), v_max=float(V.max()),
        t_mean=float(T.mean()), t_max=float(T.max()), t_rise=float(T.max() - T.min()),
        n_points=int(V.size),
    )


def main():
    kept, table = [], []
    for f in sorted(glob.glob(os.path.join(BASE, "*.mat"))):
        cell = os.path.basename(f).replace(".mat", "")
        if cell in MIXED:
            continue
        d = sio.loadmat(f)
        key = [k for k in d.keys() if not k.startswith("__")][0]
        cyc = d[key][0, 0]["cycle"]
        temp = np.nan
        tot = ok = pulses = short = highI = restY = 0
        idx = 0
        for i in range(cyc.shape[1]):
            c = cyc[0, i]
            if str(c["type"][0]) != "discharge":
                continue
            tot += 1
            try:
                temp = round(float(c["ambient_temperature"][0, 0]), 1)
            except Exception:
                pass
            ft = protocol_feats(c)
            if ft is None:
                continue
            c1 = ft["duration_s"] >= C1_MIN_DUR
            c2 = ft["mean_abs_i"] <= C2_MAX_ABS_I
            c3 = ft["rest_frac"] <= C3_MAX_REST_FRAC
            if not c1:
                short += 1
            if not c2:
                highI += 1
            if not c3:
                restY += 1
            if c1 and c2 and c3:
                idx += 1
                ok += 1
                ft.update(cell=cell, temp_c=temp, cycle_idx=idx)
                kept.append(ft)
            else:
                pulses += 1
        table.append((cell, temp, tot, ok, tot - ok, short, highI, restY))
    cols = ["cell", "temp_c", "cycle_idx", "cap_ah", "soh", "duration_s",
            "mean_abs_i", "rest_frac", "v_mean", "v_std", "v_min", "v_max",
            "t_mean", "t_max", "t_rise", "n_points"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in cols})
    print("WROTE", OUT, "rows", len(kept))
    print()
    print(f"{'cell':7s} {'T':>5s} {'放电数':>6s} {'保留':>5s} {'剔除':>5s} | {'时长短':>6s} {'电流大':>6s} {'静置多':>6s} | {'保留后SOH首->末'}")
    for cell, t, tot, ok, drop, s, h, r in table:
        sub = [k for k in kept if k["cell"] == cell]
        sub.sort(key=lambda x: x["cycle_idx"])
        if sub:
            a, b = sub[0]["soh"], sub[-1]["soh"]
            srtxt = f"{a:.3f} -> {b:.3f}" + ("  ✅单调" if b <= a + 0.02 else "  ⚠上升")
        else:
            srtxt = "—"
        print(f"{cell:7s} {t:5.0f} {tot:6d} {ok:5d} {drop:5d} | {s:6d} {h:6d} {r:6d} | {srtxt}")


if __name__ == "__main__":
    main()
