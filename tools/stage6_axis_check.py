import pandas as pd, numpy as np, sys
sys.path.append('.')
from tools.baseline.data_loader import load_session
from scipy.stats import pearsonr

df1 = load_session('S-S1')
df2 = load_session('S-S2')

print('Loaded df columns used by UKF:')
print(df1.columns.tolist())

print('\nS-S1 gz stats:')
print(df1['gz'].describe())

print('\nS-S2 gz stats:')
print(df2['gz'].describe())

print('\nKEY FINDING: UKF uses gz, but best axis for yaw is gy!')
print(f'S-S1: gz vs gy correlation: {np.corrcoef(df1["gz"].values, df1["gy"].values)[0,1]:.4f}')
