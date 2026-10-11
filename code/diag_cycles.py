# -*- coding: utf-8 -*-
"""
diag_cycles.py — 逐循环打印 type 与 Capacity，用于判定「参考满放」子集

不用判据先行：先把序列打出来看，再决定规则。
"""
import scipy.io as sio
import numpy as np
import os, sys

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
BASE = _DATA


def dump(cell, max_show=None):
    f = os.path.join(BASE, cell + ".mat")
    d = sio.loadmat(f)
    key = [k for k in d.keys() if not k.startswith("__")][0]
    cyc = d[key][0, 0]["cycle"]
    n = cyc.shape[1]
    print(f"===== {cell}  总循环 {n}")
    # 统计各类 type
    from collections import Counter
    types = Counter()
    for i in range(n):
        types[str(cyc[0, i]["type"][0])] += 1
    print("  type 计数:", dict(types))
    # 逐条打印 discharge
    rows = []
    for i in range(n):
        c = cyc[0, i]
        t = str(c["type"][0])
        if t != "discharge":
            continue
        dd = c["data"][0, 0]
        cap = float(np.asarray(dd["Capacity"]).ravel()[0]) if dd.dtype.names and "Capacity" in dd.dtype.names else np.nan
        try:
            amb = float(c["ambient_temperature"][0, 0])
        except Exception:
            amb = np.nan
        # 放电时长 = Time 的跨度
        tm = np.asarray(dd["Time"]).ravel() if "Time" in dd.dtype.names else np.array([])
        dur = float(tm[-1] - tm[0]) if tm.size > 1 else np.nan
        rows.append((i, cap, dur, amb))
    print(f"  discharge 条数 {len(rows)}")
    show = rows if max_show is None else rows[:max_show]
    print(f"  {'idx':>5s} {'cap_Ah':>8s} {'dur_s':>8s} {'amb_C':>6s}")
    for i, cap, dur, amb in show:
        print(f"  {i:5d} {cap:8.4f} {dur:8.1f} {amb:6.1f}")
    caps = np.array([r[1] for r in rows], dtype=float)
    if caps.size:
        print(f"  cap 统计: min={caps.min():.4f} p50={np.median(caps):.4f} max={caps.max():.4f}")
    print()


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["B0005", "B0033", "B0041"]
    for t in targets:
        dump(t)
