import numpy as np
import pandas as pd

def detect_stationary(df, accel_var_thresh=0.015, window_size=10):
    """
    Detects if the vehicle is stationary based purely on IMU.
    Uses rolling variance of acceleration magnitude.
    window_size = 10 (at 10Hz, this is 1 second).
    """
    if len(df) < window_size:
        return np.zeros(len(df), dtype=bool)
        
    accel_mag = df['accel_mag'].values
    
    # Calculate rolling variance
    var_accel = pd.Series(accel_mag).rolling(window=window_size, min_periods=1, center=True).var().values
    var_accel[np.isnan(var_accel)] = 0.0
    
    # Simple threshold
    is_stationary = var_accel < accel_var_thresh
    
    return is_stationary
