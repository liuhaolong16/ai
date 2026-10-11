# -*- coding: utf-8 -*-
"""NASA 电池数据探查：确认 .mat 结构，提取每循环放电容量 -> SOH 序列"""
import os
from collections import Counter
import numpy as np
from scipy.io import loadmat

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
DATA = _DATA

for name in ['B0005', 'B0006', 'B0007', 'B0018']:
    f = os.path.join(DATA, name + '.mat')
    if not os.path.exists(f):
        print(f"[{name}] 文件不存在，跳过")
        continue
    m = loadmat(f)
    keys = [k for k in m.keys() if not k.startswith('__')]
    print(f"\n=== {name} ===")
    print(f"顶层 keys: {keys}")
    cell = m[name]
    print(f"cell shape: {cell.shape}, dtype names: {cell.dtype.names}")
    if 'cycle' in cell.dtype.names:
        cycle = cell[0, 0]['cycle']
        print(f"cycle shape: {cycle.shape}, cycle dtype names: {cycle.dtype.names}")
        types = [str(cycle[0, i]['type'][0]) for i in range(cycle.shape[1])]
        print(f"cycle types 计数: {dict(Counter(types))}")
        # 提取放电容量
        caps = []
        for i in range(cycle.shape[1]):
            t = str(cycle[0, i]['type'][0])
            if t == 'discharge':
                d = cycle[0, i]['data'][0, 0]
                if 'Capacity' in d.dtype.names:
                    c = float(d['Capacity'][0, 0])
                    caps.append(c)
        caps = np.array(caps)
        print(f"放电循环数: {len(caps)}, 容量范围: [{caps.min():.3f}, {caps.max():.3f}] Ah")
        if len(caps) > 0:
            print(f"初始容量(前5循环均值): {caps[:5].mean():.3f}, 末容量: {caps[-1]:.3f}")
