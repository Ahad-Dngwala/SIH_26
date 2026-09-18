import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.baseline.data_loader import load_session

df = load_session('S-S2')
l2m = 111000
l0 = df['lat'].iloc[0]
lon2m = 111000 * np.cos(np.radians(l0))
y = (df['lat'].values - l0) * l2m
x = (df['lon'].values - df['lon'].iloc[0]) * lon2m
dy = np.zeros(len(df))
dx = np.zeros(len(df))
dt = df['time'].values
for i in range(1, len(df)-1):
    dt_c = dt[i+1] - dt[i-1]
    if dt_c > 0:
        dy[i] = (y[i+1] - y[i-1]) / dt_c
        dx[i] = (x[i+1] - x[i-1]) / dt_c
dy = pd.Series(dy).rolling(10, min_periods=1, center=True).mean().values
dx = pd.Series(dx).rolling(10, min_periods=1, center=True).mean().values

bs = dt[len(dt)//2]
be = bs + 60
m = (dt >= bs) & (dt <= be)
d_dt = df['dt'].values[m]
px = np.cumsum(dx[m] * d_dt)
py = np.cumsum(dy[m] * d_dt)
err = np.linalg.norm([px[-1] - (x[m][-1] - x[m][0]), py[-1] - (y[m][-1] - y[m][0])])
print('Direct Integration of GT_V:', err)
