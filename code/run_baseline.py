# -*- coding: utf-8 -*-
"""NASA 电池 SOH 预测 baseline：XGBoost + 留一电池交叉验证（测跨温度泛化）"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score

# --- repository-relative paths (added for the public deposit; see README.md) ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, "data")
_RESULTS = os.path.join(_REPO, "results")
_FIGURES = os.path.join(_REPO, "figures")
DATA = _DATA
# NASA 2Ah 电池：B0005=24C, B0006=4C(实际 4C 充电/室温放?), B0007=43C, B0018=25C
BATTERIES = {
    'B0005': ('B0005.mat', '24C'),
    'B0006': ('B0006.mat', '4C'),
    'B0007': ('B0007.mat', '43C'),
    'B0018': ('B0018.mat', '25C'),
}

def extract_cycles(mat_file):
    m = loadmat(mat_file)
    key = [k for k in m.keys() if not k.startswith('__')][0]
    cycle = m[key][0, 0]['cycle']
    rows = []
    for i in range(cycle.shape[1]):
        t = str(cycle[0, i]['type'][0])
        if t != 'discharge':
            continue
        d = cycle[0, i]['data'][0, 0]
        cap = float(d['Capacity'][0, 0])
        V = d['Voltage_measured'][0, 0].ravel()
        T = d['Temperature_measured'][0, 0].ravel()
        rows.append({
            'cycle_idx': len(rows) + 1,
            'capacity': cap,
            'V_mean': V.mean(), 'V_std': V.std(), 'V_min': V.min(), 'V_max': V.max(),
            'V_slope': (V[-1] - V[0]) / max(len(V) - 1, 1),
            'T_mean': T.mean(),
        })
    df = pd.DataFrame(rows)
    c0 = df['capacity'].iloc[:10].mean()
    df['soh'] = df['capacity'] / c0
    return df

# 1) 提取所有电池
frames = []
for name, (fname, temp) in BATTERIES.items():
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        print(f"[跳过] {name} 文件不存在")
        continue
    df = extract_cycles(path)
    df['battery'] = name
    df['temp_label'] = temp
    frames.append(df)
    print(f"{name} ({temp}): {len(df)} 放电循环, SOH [{df['soh'].min():.3f}, {df['soh'].max():.3f}]")

if not frames:
    print("无数据，退出")
    raise SystemExit

data = pd.concat(frames, ignore_index=True)
print(f"\n合计 {len(data)} 循环, 电池数 {data['battery'].nunique()}")

# 2) 特征 + 目标
FEATURES = ['cycle_idx', 'V_mean', 'V_std', 'V_min', 'V_max', 'V_slope', 'T_mean']
X = data[FEATURES].values
y = data['soh'].values
groups = data['battery'].values

# 3) 留一电池交叉验证（每折：留 1 个电池做测试，其余 3 个训练 → 跨温度泛化）
print("\n=== 留一电池交叉验证（跨温度泛化）===")
batteries = data['battery'].unique()
all_rmse, all_r2 = [], []
for test_b in batteries:
    tr = data[data['battery'] != test_b].index
    te = data[data['battery'] == test_b].index
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    rmse = root_mean_squared_error(y[te], pred)
    r2 = r2_score(y[te], pred)
    all_rmse.append(rmse); all_r2.append(r2)
    print(f"测试 {test_b} ({data[data.battery==test_b]['temp_label'].iloc[0]}): RMSE={rmse:.4f}, R2={r2:.4f}")

print(f"\n平均 RMSE={np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}, 平均 R2={np.mean(all_r2):.4f}")

# 4) 特征重要性
model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X, y)
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n特征重要性:")
print(imp.to_string())
