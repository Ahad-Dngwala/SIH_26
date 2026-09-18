# Stage 1: Low-Speed Ground-Truth & Data-Quality Audit
**Date:** September 16, 2026

## Executive Summary
This audit rigorously investigated the low-speed failure of Channel A and B models (<15 m/s, where RMSE degrades >5 m/s). 
The diagnosis is clear: **The ground-truth labels are clean, but the smartphone input signal (S-Dataset) is heavily undersampled and degraded by naive interpolation.**

### Audit Findings

#### 1. Ground Truth Provenance
- **Current Ground Truth:** The pipeline (`iovnbd_common.py` -> `03_window.py`) correctly traces the `labels.npy` target directly to the `Velocity (km/hr)` column in the `V-Dataset`. 
- **Quality:** A low-speed binning analysis on >200,000 samples confirmed that the GPS velocity is incredibly clean. During 29,163 known stationary samples (verified via OBD wheel speed), the mean GPS speed was exactly 0.026 m/s with a standard deviation of 0.043 m/s. Only 0.08% of stationary samples falsely reported speeds >0.5 m/s. 
- **Verdict:** The labels do NOT require cleaning. They are highly trustworthy.

#### 2. The Real Problem: S-Dataset Undersampling
- The `S-Dataset` (smartphone IMU data) was assumed to be 100 Hz. 
- Parsing the raw timestamp column (`TIME SINCE START (ms)`) reveals an **empirical sample rate of exactly 17.17 Hz**. 
- A nominal 5-second window only contains ~85 real hardware samples.
- **The Interpolation Trap:** `01_resample.py` uses linear interpolation to forcefully up-sample this 17 Hz signal to 100 Hz (stretching 85 samples to 500 samples). At highway speeds, the vibration amplitude is large enough that the 17 Hz envelope captures the energy. However, below 15 m/s, the true vibration frequency of the car exceeds the Nyquist limit of the 17 Hz sensor, and the linear interpolation effectively flattens and destroys whatever micro-vibration signal existed. 

## Final Directives (As Requested)

1. **Current ground-truth signal used by the model:** 
   - GPS-derived `Velocity (km/hr)` from the V-Dataset.
2. **Recommended ground-truth signal for future training:** 
   - Continue using the GPS velocity. It is definitively clean even down to 0 m/s.
3. **Should 100 Hz interpolation be retained?**
   - **NO.** The 17 Hz signal is being destroyed by linear up-sampling. The models should be modified to accept variable-length raw sequences (e.g., using LSTMs/Transformers without resampling), or the interpolation method must be changed to preserve energy (e.g., FFT-based resampling) rather than linear smoothing.
4. **Should low-speed samples be hard-labeled, cleaned, masked, or treated probabilistically?**
   - Because the 17 Hz smartphone sensor physically lacks the bandwidth to capture low-speed vibration, the ML models cannot physically learn this mapping. The UKF must **mask/treat probabilistically** any ML predictions when the Map/Wheel-speed estimates drop below 15 m/s.
5. **Should further model training proceed?**
   - **NO.** Retraining the same architectures on heavily interpolated 17 Hz data will continue to fail. Stop model training and wire the Trust-Aware UKF logic.

## Attachments
- **Code:** Available in `tools/audit/`
- **Plots:** `reports/plot_A_stationary.png`, `reports/plot_D_low_speed.png` (Overlapping clean GPS vs. flattened interpolated IMU).
