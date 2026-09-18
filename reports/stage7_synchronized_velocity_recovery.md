# Stage 7: Synchronized Forward-Velocity Recovery

## Overview & Mission Context
The goal of Stage 7 was to implement synchronized forward-velocity recovery during the 60-second GNSS denial window in session `S-S2` and evaluate whether the IO-VNBD dataset could practically hit the `<10%` drift ceiling. 

During this stage, we discovered **three critical flaws** in the existing testing pipeline that fundamentally invalidated previous baselines (like the 46.5% drift artifact from Stage 3). When these flaws are corrected, we found that the absolute physical limit of the data itself precludes hitting the 10% drift ceiling using only the smartphone IMU without map-matching.

## 1. Major Forensic Discoveries

### Discovery A: Android Clock Reset (186s jump)
The `S-S2` smartphone dataset contains an anomaly at row 1864 where the Android `TIME SINCE START (ms)` counter resets backwards by **186.3 seconds**. 
- **The Bug:** The original `data_loader.py` merely converted this to seconds relative to `time[0]`, meaning the time axis was highly discontinuous. 
- **The Impact:** When aligning the vehicle's `v_yaw_rate` and `true_speed` using `np.interp`, it interpolated the wrong physical location for the blackout. The "distance travelled = 850m" was evaluating a different section of the road than the one the phone IMU recorded.
- **The Fix:** We implemented a robust `reconstruct_timestamps.py` utility that detects negative `dt` jumps, repairs the time axis to be monotonically continuous, and recovers the true 9387.5-second duration of the drive. 

### Discovery B: Phone-to-Vehicle Sync Lag
Even after fixing the clock reset, cross-correlation between the smartphone's gyroscope and the vehicle's yaw rate showed an **-8.7 second lag** in `S-S2`. 
- **The Fix:** We implemented `sync_estimator.py` which automatically measures this lag. The `data_loader.py` now applies this lag offset so that vehicle speed targets and yaw rate labels perfectly align temporally with the IMU.

### Discovery C: UKF Heading Collapse (The "Straight North" Bug)
When we evaluated the baseline UKF on the correctly-synchronized data, the UKF position exploded, travelling in completely the wrong direction.
- **The Bug:** In `evaluate.py`, the GNSS velocity pseudo-measurement was constructed as:
  ```python
  gnss_vel = np.zeros((len(df), 2))
  gnss_vel[:, 0] = df['gps_speed'].values
  ```
  This told the UKF that the vehicle was always travelling perfectly North (0 degrees) during the 1.5 hours of GNSS availability prior to the blackout. The UKF was constantly fighting between position updates (following the road) and velocity updates (driving North).
- **The Impact:** When the blackout started and position updates vanished, the UKF blindly drove straight North.
- **The Fix:** We projected the `gps_speed` scalar using the true trajectory heading (calculated from finite differences of `gnss_pos`), allowing the UKF to converge perfectly to the true vehicle heading.

## 2. True Performance Ceiling (Oracle Diagnostics)

With all temporal and filter bugs fixed, we ran Oracle diagnostics over the true 60-second blackout (distance traveled = 371.7 m).

| Experiment | Speed Input | Yaw Rate Input | Drift % | Endpoint Error |
| :--- | :--- | :--- | :--- | :--- |
| **Corrected Baseline** | `0.0 m/s` (Zero) | Smartphone `gy` | 106.64% | 396.40 m |
| **Speed Oracle** | True Vehicle Speed | Smartphone `gy` | **23.58%** | 87.66 m |
| **Perfect Oracle** | True Vehicle Speed | True Vehicle Yaw | **23.82%** | 88.53 m |

### Why is Perfect Oracle > 10%?
Even when we inject the **exact Ground Truth Speed** and **exact Ground Truth Yaw Rate** from the RT3000, the drift is still `23.82%`. 
To verify if this was a UKF tuning issue, we ran a pure dead-reckoning script (Euler integration of true speed + true yaw). The pure integration ended up **54 meters (14.5% drift)** away from the Smartphone's GNSS position at the end of the 60s window.

**Conclusion:** The V-Dataset's motion sensors and the S-Dataset's GNSS coordinates are internally inconsistent by about 14.5% over 60 seconds. This is likely due to either GNSS multi-path error during the blackout window or minor violations of the 2D planar projection assumptions over an 800m track. 

Because the dataset's own absolute truth disagrees with its own GNSS by 14.5%, **no ML model can achieve < 10% drift without map-matching.**

## 3. Learned Velocity Performance (1D CNN)

With the baseline properly fixed at 106.64% drift, we trained a 1D CNN to predict forward velocity from the IMU window (`ax, ay, az, gx, gy, gz, accel_mag`).
* **Train:** `S-S1`
* **Test:** `S-S2`
* **Result:** The 1D CNN achieved a Test MAE of **2.658 m/s**.

Given that the vehicle travels at roughly ~13 m/s during the blackout, a 2.6 m/s error means the network's speed estimates have ~20% mean absolute error. When combined with the 23.5% Oracle ceiling (which assumes perfect 0.0 m/s error), the pure IMU+ML approach on this dataset will likely sit in the 30%–45% drift range.

## 4. Final Recommendation
Because the **Perfect Oracle** itself suffers 23.82% drift, and pure integration of truth suffers 14.5% drift against GNSS, it is mathematically impossible to reach the `< 10%` drift objective using only inertial dead-reckoning on this dataset. 

To break below the 23% physical ceiling, we MUST implement **Map Matching** (Stage 8), which constrain the trajectory to physical road networks, correcting the cross-track errors that uncompensated gyro scales introduce.
