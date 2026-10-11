# -*- coding: utf-8 -*-
"""
extract_features.py — NASA 电池数据特征提取
把 data/*.mat 逐放电循环提取为一维表，存 features.csv

设计纪律：
- 温度取自 .mat 内 ambient_temperature，不靠文件名（已在 06-数据采集记录 中实测确认）
- 混合温度电池（同电池内出现两个温度值）默认排除，理由见 06-数据采集记录 §四
- 只做提取，不做建模；提取结果落 CSV，后续建模脚本读同一个 CSV
"""
import scipy.io as sio
import numpy as np
import glob, os, csv, json

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
DATA = _DATA
OUT = os.path.join(_DATA, "features.csv")
NOMINAL_AH = 2.0

# 混合温度电池：同电池内出现两个温度值（实测：B0038-40 = 24+44C, B0042-44 = 4+22C）
MIXED = {"B0038", "B0039", "B0040", "B0042", "B0043", "B0044"}


def cell_of(cyc):
    """按【字段名】取，不按位置取。

    实测 data 是结构化数组，字段为：
        Voltage_measured, Current_measured, Temperature_measured,
        Current_load, Voltage_load, Time, Capacity
    （★ 注意：共 7 个字段；早期版本按位置取会把 Capacity 取成 Time，静默产生 cap=0）
    """
    d = cyc["data"][0, 0]
    names = d.dtype.names
    g = lambda n: np.asarray(d[n]).ravel()
    return (g("Voltage_measured"), g("Current_measured"), g("Temperature_measured"),
            g("Voltage_load"), g("Time"), g("Capacity"), names)


def feats_from_discharge(cyc):
    """从单个放电循环提取物理特征"""
    V, I, T, Vload, Tsec, Cap, names = cell_of(cyc)
    if V.size < 10:
        return None
    # 放电容量：字段 Capacity（Ah）。实测为长度 1 的数组
    if Cap.size < 1:
        return None
    cap_ah = float(Cap.ravel()[0])
    if not np.isfinite(cap_ah) or cap_ah <= 0:
        return None
    dur = float(Tsec[-1] - Tsec[0])
    if dur <= 0:
        return None
    # 电压特征
    v_mean, v_std = float(V.mean()), float(V.std())
    v_min, v_max = float(V.min()), float(V.max())
    # IR 降（起始若干点的最大瞬时跌落）
    k = min(8, V.size - 1)
    ir_drop = float(V[:k].max() - V[k]) if k >= 2 else np.nan
    # 电压平台斜率：去掉首尾 10% 后的线性斜率（V/s）
    lo, hi = int(V.size * 0.10), int(V.size * 0.90)
    slope = np.nan
    if hi - lo >= 5:
        xs = np.asarray(Tsec[lo:hi], dtype=float)
        ys = np.asarray(V[lo:hi], dtype=float)
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.sum() >= 5:
            xs2, ys2 = xs[m], ys[m]
            if (xs2[-1] - xs2[0]) > 0:
                try:
                    slope = float(np.polyfit(xs2 - xs2[0], ys2, 1)[0])
                except np.linalg.LinAlgError:
                    slope = np.nan
    # 温度特征（电池本体温度，非环境）
    t_mean, t_max = float(T.mean()), float(T.max())
    t_rise = float(T.max() - T.min())
    # 电流
    i_mean, i_std = float(I.mean()), float(I.std())
    return dict(
        cap_ah=cap_ah, soh=cap_ah / NOMINAL_AH,
        duration_s=dur,
        v_mean=v_mean, v_std=v_std, v_min=v_min, v_max=v_max,
        ir_drop=ir_drop, v_slope=slope,
        t_mean=t_mean, t_max=t_max, t_rise=t_rise,
        i_mean=i_mean, i_std=i_std,
        n_points=int(V.size),
    )


def main():
    rows = []
    summary = {}
    for f in sorted(glob.glob(os.path.join(DATA, "*.mat"))):
        cell = os.path.basename(f).replace(".mat", "")
        if cell in MIXED:
            summary[cell] = "SKIP(mixed-temp)"
            continue
        try:
            d = sio.loadmat(f)
            key = [k for k in d.keys() if not k.startswith("__")][0]
            cyc = d[key][0, 0]["cycle"]
        except Exception as e:
            summary[cell] = f"ERR load {e}"
            continue
        amb_all, n_dis = [], 0
        idx = 0
        for i in range(cyc.shape[1]):
            c = cyc[0, i]
            if str(c["type"][0]) != "discharge":
                continue
            n_dis += 1
            try:
                amb = round(float(c["ambient_temperature"][0, 0]), 1)
            except Exception:
                amb = np.nan
            amb_all.append(amb)
            ft = feats_from_discharge(c)
            if ft is None:
                continue
            idx += 1
            ft.update(cell=cell, temp_c=amb, cycle_idx=idx)
            rows.append(ft)
        temps = sorted(set(amb_all))
        summary[cell] = f"discharge={n_dis} temps={temps}"
    # 写 CSV
    if not rows:
        print("NO ROWS")
        return
    cols = ["cell", "temp_c", "cycle_idx", "cap_ah", "soh", "duration_s",
            "v_mean", "v_std", "v_min", "v_max", "ir_drop", "v_slope",
            "t_mean", "t_max", "t_rise", "i_mean", "i_std", "n_points"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print("WROTE", OUT)
    print("ROWS", len(rows))
    print("CELLS", len({r["cell"] for r in rows}))
    by_temp = {}
    for r in rows:
        by_temp.setdefault(r["temp_c"], set()).add(r["cell"])
    print("TEMP -> cells:", {k: len(v) for k, v in sorted(by_temp.items())})
    print("\nPER-CELL SUMMARY")
    for k in sorted(summary):
        print(f"  {k}  {summary[k]}")


if __name__ == "__main__":
    main()
