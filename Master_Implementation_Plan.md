# Master Implementation Plan
## AI/ML Intelligent Dead Reckoning System - SIH Problem Statement 26168

This document is the single source of truth for building the system end to end. It is written so that it can be handed to other Claude instances, each of which will take one section and turn it into working code with their team member. Read the whole thing once before starting your own section, because the modules depend on each other's input/output contracts defined here.

If you are a Claude instance picking up one task from this plan: find your task in Section 12, then read the matching detailed section earlier in the document (do not skip straight to Section 12, the contracts and specs live in the earlier sections).

---

## 0. System Summary (read this first)

We are building a phone-based dead reckoning system that keeps working when GPS drops. It has two deliverables that share one codebase:

1. An Android app (uses the phone's own IMU).
2. An edge engine (same models, runs on a Raspberry Pi / Jetson class board reading an external IMU).

The system is made of 6 independent models plus 2 non-ML subsystems (Kalman fusion core, map matching). Each model is small, trained separately, and exported to ONNX/TFLite for on-device inference. Nothing runs in the cloud at inference time. Cloud/desktop is only used for offline training.

Core idea in one line: instead of trusting one IMU integration pipeline, we run two independent speed estimators plus a road-vibration fingerprint classifier, and fuse all of it with GPS in a single Kalman filter, so that when one channel fails the others catch it.

---

## 1. Repository Structure

Use one monorepo. Suggested layout:

```
idr-system/
  data/                                              [Layer 1 - Data & Models]
    raw/                  (IO-VNBD downloads, untouched)
    processed/            (windowed, normalized tensors, per model)
    scripts/               (download + preprocessing scripts)
  models/                                            [Layer 1 - Data & Models]
    alignment_net/
    channel_a_velocity/
    channel_b_velocity/
    road_signature/
    calibration_adapter/
    common/                (shared dataset classes, normalization utils)
  fusion_core/                                       [Layer 2 - Fusion & Map]
    python_prototype/      (UKF prototype, filterpy or custom)
    cpp/                    (production C++ port, Eigen based)
  map_matching/                                      [Layer 2 - Fusion & Map]
    osm_extraction/
    hmm/
  android_app/                                       [Layer 3 - Platform & Edge]
    app/                    (Kotlin, Jetpack Compose)
    native/                 (NDK/C++ bridge to fusion_core/cpp)
  edge_engine/                                       [Layer 3 - Platform & Edge]
    src/                    (C++ main loop, sensor drivers)
  tools/
    benchmark_replay/       (offline trajectory replay + drift plots)     [Layer 2]
    export/                 (ONNX -> TFLite conversion scripts)          [Layer 1]
  docs/
    this file, plus per-module READMEs
```

Ownership is a starting point, not a wall. Layer 2 will read `models/common/` to know the exact tensor shapes it receives, and Layer 3 will read `fusion_core/cpp/` headers to know what it's linking against. Nobody works in a silo, the folder boundaries just say who is responsible for keeping that folder green.

Every model folder must contain: `train.py`, `model.py`, `dataset.py`, `export.py`, `config.yaml`, and a `README.md` with input/output shapes and current best metric. This is not optional, it is how other team members and other Claude sessions will know what state a module is in without reading all the code.

---

## 2. Environment and Tooling

- Python 3.11, PyTorch 2.x, PyTorch Lightning for training loops, Weights and Biases (or plain CSV logging if W&B is not available) for experiment tracking.
- Export path: PyTorch -> ONNX (opset 17) -> ONNX Runtime Mobile for Android, or -> TFLite via onnx2tf if a team member prefers TFLite tooling. Pick ONNX Runtime Mobile as the default unless a specific model has TFLite-only ops.
- Quantization: post-training dynamic int8 quantization for every deployed model, using ONNX Runtime's quantization tool.
- Android: Kotlin, Jetpack Compose, min SDK 26 (Android 8.0), target latest stable SDK. NDK for the native fusion core bridge.
- C++: C++17, Eigen for linear algebra in the UKF, CMake as the build system, shared between android_app/native and edge_engine/src.
- Map data: OSM extracts via osmium-tool or the osmnx Python package, GraphHopper-style graph structure for routing/adjacency.
- Version control: git, one branch per module during development, merge to main only after that module's own test script passes.

---

## 3. Data Pipeline (Phase 0) [Layer 1 - Data & Models]

If you are joining Layer 1 partway through: this section and Section 4 are the whole job. You do not need to read Section 5 onward to be useful here, though it helps to know that Sections 5 and 7 downstream are waiting on the exact tensor shapes and `norm_stats.json` files this section produces, so changing an interface here after Layer 2/3 have started consuming it is expensive. Check `data/processed/README.md` (produced by 3.2 step 4) before assuming the current state, it will tell you exactly what has been run and with what stats.

### 3.1 Dataset

Primary: IO-VNBD (Onyekpe et al., Coventry University). Download both the Synchronised and Unsynchronised V and S folders (V = vehicle-extracted ground truth, S = smartphone-recorded).

Secondary (collected by the team before the finale): short GNSS-available loops on at least one two-wheeler, one auto-rickshaw, and one car, each 10-20 minutes, logging raw phone IMU + GNSS at the phone's native max rate. This is what seeds the road-signature packs and the calibration adapter, since IO-VNBD has no Indian two-wheeler data.

### 3.2 Preprocessing steps (apply in this order)

1. Resample every stream to a common 100 Hz using linear interpolation on timestamps. Do not use nearest-neighbor resampling, it introduces step artifacts that models will latch onto as fake features.
2. Time-align the S (smartphone) and V (vehicle ground truth) streams using the timestamp offset given in the dataset metadata. For the Unsynchronised split, do NOT try to force alignment, keep it as noise-augmentation data only (see 3.4).
3. Window the data. Standard window: 2 seconds (200 samples at 100 Hz), 50 percent overlap (stride 100 samples). Channel B uses a 4 second window (400 samples) per the CarSpeedNet-style approach, still 50 percent overlap.
4. Normalize per channel (z-score), computing mean/std on the TRAIN split only, then applying the same stats to val/test. Save the stats to a `norm_stats.json` per model so the same values are baked into the on-device preprocessing code later.
5. Split by route/session, not by random window. A route's windows must all go to the same split (train, val, or test). 70 percent of sessions to train, 15 percent val, 15 percent test. Splitting by random window instead of by route will leak information and give you fake high accuracy that collapses in the field.

### 3.3 Labels needed per model

- Alignment net: pitch/roll/yaw offset. If IO-VNBD does not give this directly, derive a pseudo-label from the gravity vector during stationary/near-constant-velocity segments plus GNSS course-over-ground as the yaw reference.
- Channel A and Channel B: forward velocity scalar per window, taken from the V (vehicle ground truth) stream.
- Road-signature classifier: segment ID, derived by dividing each corridor's route into fixed-length arc segments (e.g. 50 to 100 meters each) using the GNSS ground truth positions, then labeling every window with the segment its GNSS timestamp falls into.
- Calibration adapter: same as Channel A/B (velocity), but drawn only from the short calibration-drive session for a specific vehicle.

### 3.4 Augmentation

- Random small time-shift jitter (+/- 5 percent of window length).
- Additive Gaussian noise scaled to each sensor's known noise floor (check IMU datasheet noise density, do not guess).
- Random gain/bias perturbation to simulate a slightly different phone/vehicle (helps the calibration adapter generalize faster later).
- The Unsynchronised IO-VNBD split is used specifically as a "noisy labels" augmentation source for Channel A/B training, to make those models robust to the same kind of centre-of-gravity offset error the dataset documentation itself flags.

---

## 4. Model Specifications (Phase 1) [Layer 1 - Data & Models]

Every model below is small on purpose. On-device latency budget is tight (see Section 9), so do not scale these up without re-checking the latency budget. Each subsection below is independently buildable once Section 3's processed tensors exist, so within Layer 1 the two people here can split by subsection (e.g. one owns 4.1/4.2, the other owns 4.3/4.4, then both converge on 4.5/4.6) rather than working sequentially.

### 4.1 Alignment / Mount Calibration Net (Layer 1)

- Purpose: estimate the phone's pitch, roll, yaw relative to the vehicle's direction of travel.
- Input: 2 second window, 9 channels (accel x/y/z, gyro x/y/z, mag x/y/z), shape (200, 9).
- Architecture: Conv1d(9 to 32, kernel 5, stride 1) - ReLU - Conv1d(32 to 64, kernel 5, stride 2) - ReLU - Conv1d(64 to 64, kernel 3, stride 2) - ReLU - GlobalAveragePool - FC(64 to 32) - ReLU - FC(32 to 3).
- Output: 3 values, pitch/roll/yaw offset in radians.
- Loss: MSE, with yaw handled as sin/cos pair to avoid the wraparound discontinuity (predict 4 values total: pitch, roll, sin(yaw), cos(yaw), then reconstruct yaw with atan2).
- Optimizer: Adam, lr 1e-3, cosine annealing over training.
- Batch size: 128.
- Epochs: 60, early stopping on val loss with patience 10.
- Target: mean angular error under 3 degrees on held-out routes.

### 4.2 Channel A - NHC-constrained bias-correction network (Layer 1)

- Purpose: correct accelerometer/gyro bias and scale error so that integrated velocity stays close to true velocity.
- Input: 2 second window, 6 channels (accel x/y/z, gyro x/y/z), shape (200, 6).
- Architecture: Temporal Convolutional Network (TCN), 4 dilated causal conv blocks, dilations 1/2/4/8, channels 32/64/64/128, kernel size 3, residual connections between blocks, followed by GlobalAveragePool - FC(128 to 64) - ReLU - FC(64 to 1).
- Output: forward velocity scalar (m/s) for the window's end timestamp.
- Loss: Huber loss (delta 1.0), more robust to the labeling noise in the Unsynchronised augmentation data than plain MSE.
- Optimizer: Adam, lr 1e-3, ReduceLROnPlateau (factor 0.5, patience 8) on val loss.
- Batch size: 64.
- Epochs: 100, early stopping patience 15.
- Dropout: 0.2 after each conv block.
- Gradient clipping: max norm 5.
- Target: velocity RMSE under 0.5 m/s on held-out routes (synchronized split).
- Note: NHC (no lateral slip, no vertical velocity) is enforced downstream in the UKF process model, not inside this network. Keep this network a pure regressor.

### 4.3 Channel B - accelerometer-only velocity net (CarSpeedNet-style) (Layer 1)

- Purpose: independent, gyro-free forward speed estimate, used as a cross-check against Channel A.
- Input: 4 second window, 3 channels (accel x/y/z only), shape (400, 3).
- Architecture: 4 Conv1d blocks, channels 32/64/128/128, kernel 7, stride 2 each (so the sequence length shrinks by 16x total), BatchNorm + ReLU after each conv, GlobalAveragePool - FC(128 to 32) - ReLU - FC(32 to 1).
- Output: forward velocity scalar (m/s).
- Loss: MAE.
- Optimizer: Adam, lr 1e-3, cosine schedule.
- Batch size: 128.
- Epochs: 90, early stopping patience 12.
- Target: velocity RMSE around 1.5 to 2.0 m/s (this channel is deliberately less accurate than Channel A, its job is disagreement detection, not primary accuracy).

### 4.4 Road-Signature Segment Classifier (Layer 1)

- Purpose: classify which pre-mapped road segment the vehicle is currently on, from vibration signature alone, to use as a periodic drift-reset anchor.
- Input: 2 second window, 6 channels (accel + gyro), shape (200, 6).
- Architecture: 3 Conv1d blocks, channels 32/64/128, kernel 5, MaxPool(2) after each block, GlobalAveragePool - FC(128 to N_segments).
- Output: softmax over N_segments (N depends on corridor length, roughly 1 segment per 50-100 meters of mapped route).
- This model is trained per-corridor or per-city, not globally. Ship it as a separate small "signature pack" per region, a few MB each, downloaded/cached only for corridors the user actually drives.
- Loss: cross-entropy with label smoothing 0.1.
- Optimizer: Adam, lr 1e-3.
- Batch size: 64.
- Epochs: 80, early stopping patience 10.
- Target: top-1 segment accuracy above 90 percent on held-out passes of the same corridor. Accuracy alone is not sufficient here since segments are imbalanced by corridor length (a long straight segment gets far more windows than a short one) - report macro-averaged recall and macro-averaged F1 alongside accuracy for every evaluation run, and treat a class with recall well below the macro average as a real problem even if overall accuracy looks fine.
- Deployment rule: only trigger a drift-reset when softmax max probability exceeds 0.85, to avoid false resets from a low-confidence guess.

### 4.5 Per-Vehicle Calibration Adapter (Layer 1)

- Purpose: fast on-device recalibration of Channel A and Channel B for a specific vehicle and mount, using a short GNSS-available calibration drive (10 to 30 seconds).
- Method: LoRA-style low-rank adapters inserted into the final FC layers of Channel A and Channel B. Rank 4 to 8. Trainable parameter count must stay under 50k total across both models combined.
- Base models (Channel A, Channel B) are frozen during this step. Only the adapter weights train.
- Training: on-device, plain SGD (not Adam, to keep the optimizer state small), lr 1e-4, 5 to 10 epochs over the short calibration session, batch size as large as the calibration session allows (likely 8 to 16 given the short duration).
- Label source: GNSS-derived velocity during the calibration drive.
- Output: a small adapter weight file (a few KB) saved per vehicle profile, loaded alongside the base models at inference time.

### 4.6 Export and Quantization (applies to all 5 models above) (Layer 1)

1. Export each trained PyTorch model to ONNX, opset 17, with fixed input shape (batch=1) for mobile.
2. Run ONNX Runtime's post-training dynamic quantization (int8) on each exported model.
3. Verify quantized accuracy has not dropped more than 5 percent relative to the float32 model on the val set. If it has, fall back to float16 quantization for that specific model instead of int8.
4. Save quantized models under a fixed naming scheme: `alignment_net.onnx`, `channel_a.onnx`, `channel_b.onnx`, `road_signature_<region>.onnx`.
5. Combined footprint target: under 8 MB for the four core models (excluding per-region road-signature packs, which are downloaded separately).

### 4.7 Evaluation Metrics Standard (applies to every classification-style output in this system)

Accuracy alone is judged insufficient for anything the judges will scrutinize or that feeds a safety-relevant decision downstream (the road-signature classifier's output directly triggers a position reset in Section 5.3). For every classifier - currently the road-signature classifier (4.4), and the GNSS quality classifier if it is ever upgraded from rule-based to learned (5.5.4) - report all of the following on every evaluation run, not accuracy in isolation:

- Accuracy (overall correct / total)
- Macro-averaged recall (catches a class the model is quietly ignoring even while overall accuracy stays high)
- Macro-averaged F1 (the single number to track run over run for regressions)

Regression models (Channel A, Channel B, alignment net, calibration adapter) keep their existing RMSE/MAE/angular-error targets, those are the correct metric family for continuous outputs, recall and F1 do not apply there.

---

## 5. Fusion Core - Unscented Kalman Filter (Phase 2) [Layer 2 - Fusion & Map]

### 5.1 State vector

7 states: `[pn, pe, vn, ve, psi, ba, bg]`

- pn, pe: position north/east in meters, local tangent frame.
- vn, ve: velocity north/east in m/s.
- psi: heading in radians.
- ba: residual accelerometer bias.
- bg: residual gyro bias.

### 5.2 Process model

Constant-velocity, constant-turn-rate kinematic model, propagated at the filter's update rate (10 Hz on phone, up to 200 Hz on edge engine). Process noise covariance Q tuned separately for the "GNSS available" regime (tighter, since we trust GNSS to catch drift) and the "blackout" regime (looser on position, tighter on the bias terms, since we are relying more on the IMU-derived channels).

### 5.3 Measurement/update sources (this is the part that differs from a standard UKF)

Four independent evidence sources feed the innovation step, not the usual two:

1. GNSS position and velocity, when available. Standard measurement noise R based on reported GNSS accuracy/HDOP.
2. Channel A velocity estimate (forward speed, rotated into north/east using the current heading estimate).
3. Channel B velocity estimate (same rotation), with its measurement noise R set higher than Channel A's under normal conditions.
4. Road-signature anchor position (the segment midpoint), applied as a soft position correction only when the classifier's confidence exceeds 0.85, with a measurement noise R sized to roughly half the segment length.

### 5.4 Trust weighting logic (the "learned, context-dependent weighting" from the architecture doc)

Before each update step, compute the disagreement between Channel A and Channel B's velocity estimates. If the disagreement exceeds a threshold (start with 1.5x Channel B's own typical RMSE, tune from validation data), inflate Channel A's measurement noise R for that cycle only, so the filter leans more on Channel B and the road-signature anchor instead. This is a simple gain-scheduling rule, not a learned network, keep it that way for the first version, it is easier to debug and defend to judges.

### 5.5 Implementation plan

Do not wait on Layer 1's trained models to start this section. The filter's process model, sigma-point math, and 4-source update logic can be built and unit-tested (Section 11.2) against synthetic sensor data and ground-truth-derived velocity with injected noise standing in for Channel A/B, entirely independent of whether the real models exist yet. Layer 1's real model outputs are only needed for the final validation step below (checking real drift numbers against IO-VNBD), which is a later checkpoint, not the starting point.

1. Prototype in Python first, using `filterpy`'s UnscentedKalmanFilter class or a custom implementation if filterpy's API does not fit the 4-source update cleanly. Validate against the IO-VNBD held-out routes using the benchmark replay tool (Section 8).
2. Once the filter logic and tuning (Q, R matrices, thresholds) are locked, port to C++ using Eigen. This becomes a shared library (`fusion_core`) linked by both the Android native bridge and the edge engine.
3. Sigma point parameters: alpha = 1e-3, beta = 2, kappa = 0 (standard starting values, tune alpha if the filter is numerically unstable).
4. GNSS quality classifier for hand-off timing: a simple rule-based check first (satellite count, HDOP, sudden position jump detection), upgrade to a small learned classifier only if the rule-based version proves insufficient during testing. Target hand-off to INS-only mode under 200 milliseconds from GNSS loss detection.
5. Re-admission of GNSS after reacquisition uses a ramped trust weight over roughly 2 to 3 seconds (linearly increase GNSS's measurement trust from near-zero to full), so the position estimate does not visibly snap when GNSS comes back after a long blackout.

---

## 6. Map Matching (Phase 3) [Layer 2 - Fusion & Map]

1. Extract OSM data for target corridors using osmium-tool or osmnx, at build time (not at runtime), and store as a local routing graph (nodes = intersections/shape points, edges = road segments with associated geometry).
2. Implement an HMM over candidate road edges: emission probability based on perpendicular distance from the fused position to each candidate edge (Gaussian, sigma tuned to GNSS/fusion uncertainty), transition probability based on routing-graph connectivity and distance between candidate edges across consecutive timesteps.
3. Decode with Viterbi over a sliding window of the last 10 to 20 fused position updates, re-running the decode every cycle but only committing the earliest window position (this avoids the decoded path jittering as new evidence arrives).
4. Output: snapped position (lat/lon) plus the matched road edge ID, both consumed by the UI layer and by the road-signature module (to confirm which corridor's signature pack should be active).

---

## 7. Android App (Phase 4) [Layer 3 - Platform & Edge]

### 7.1 Sensor ingest

- Use Android's raw `SensorManager` callbacks for `TYPE_ACCELEROMETER`, `TYPE_GYROSCOPE`, `TYPE_MAGNETIC_FIELD`, requested at `SENSOR_DELAY_FASTEST` or a custom sampling period matching 100 Hz.
- Do NOT use `TYPE_LINEAR_ACCELERATION` or any other fused/virtual sensor, since it applies an undocumented vendor filter that competes with our own models.
- Run the sensor callback on a dedicated background thread, not the UI thread.
- Maintain a ring buffer (native code, C++ via NDK) so the JVM garbage collector never touches the hot path.
- Apply a FIR low-pass pre-filter in native code before any window is handed to a model (cutoff tuned to remove above roughly 20-30 Hz road noise while keeping vehicle dynamics intact, confirm exact cutoff empirically against IO-VNBD).

### 7.2 App structure

- Kotlin, Jetpack Compose for UI.
- Foreground service with a persistent notification for continuous background sensor logging and inference (required for Android to keep sampling while the screen is off or another app is in front).
- ONNX Runtime Mobile for model inference, one `OrtSession` per model, kept warm (not recreated per inference call).
- JNI bridge to the shared `fusion_core` C++ library for the UKF step.
- Room database (SQLite) for on-device logging of raw windows used later for calibration/retraining, opt-in only.
- MapLibre GL (or Mapbox Navigation SDK in offline tile mode) for the navigation UI, rendering the vehicle icon from the map-matched position stream.

### 7.3 Runtime loop (per cycle, target under 40 ms end to end)

1. Pull latest window from ring buffer.
2. Run Alignment net (only periodically, e.g. once every few seconds, not every cycle, since mount angle does not change fast).
3. Run Channel A and Channel B in parallel (can run sequentially on CPU if the device lacks a capable NPU/GPU delegate, both are small enough).
4. Compute Channel A/B disagreement, adjust R as per Section 5.4.
5. If due, run road-signature classifier, check confidence threshold.
6. Feed GNSS (if available), Channel A, Channel B, and road-signature anchor (if triggered) into the UKF update step via JNI.
7. Pass the UKF's fused position into the map-matching module.
8. Update the UI with the snapped position.

### 7.4 Calibration flow (user-facing)

1. On first use with a new vehicle, prompt the user to start a "calibration drive" while GNSS is available.
2. Log 10 to 30 seconds of raw IMU + GNSS.
3. Run the on-device adapter fine-tune (Section 4.5) in the background, show a simple progress indicator.
4. Save the resulting adapter under a named vehicle profile the user can switch between later.

---

## 8. Edge Engine (Phase 5) [Layer 3 - Platform & Edge]

- Same `fusion_core` C++ library, same ONNX models, different sensor ingest and I/O layer.
- Sensor ingest: UART or SPI driver reading a FOG-grade IMU at roughly 200 Hz, normalized into the same windowed-tensor format used by the phone app before touching any model (this shared normalization step is what keeps the two deliverables one codebase).
- Output: no navigation UI. Instead, stream the fused position (and optionally the map-matched edge ID) as JSON or a simple binary struct over a serial/socket interface, so a host system (robotics stack, fleet computer) can consume it.
- Reference hardware target: Raspberry Pi CM4 or Jetson Orin Nano class board, running ONNX Runtime's C++ API for inference.

---

## 9. Benchmark Replay Tool (build this early, use it constantly) [Layer 2 - Fusion & Map]

Build this in parallel with Layer 1's model training, not after it, using random/dummy values in place of Channel A/B/road-signature outputs until real ones exist. Swap in real model outputs as Layer 1 delivers them, checkpoint by checkpoint.

Build a small offline tool (Python) that:

1. Takes a held-out IO-VNBD route (or a collected route).
2. Simulates a GNSS blackout over a chosen window of the route (masking GNSS input to the fusion core for that window).
3. Runs the full Python-prototype pipeline (Channel A, Channel B, road-signature, UKF, map-matching) over the route offline.
4. Plots three trajectories on the same basemap: raw double-integration dead reckoning, our fused output, and ground truth.
5. Reports drift as a percentage of distance traveled during the blackout window, matching ISRO's own benchmark definition exactly.

This tool is the single most important thing to have working early, since every model and every fusion tuning decision should be checked against it before moving on. It is also the screening-stage deliverable.

---

## 10. Latency, Memory, and Battery Budgets

- End-to-end pipeline latency target: under 40 ms per cycle on a mid-range Android SoC, comfortably inside the 100 ms period needed for 10 Hz output.
- Combined on-device model footprint: under 8 MB (core 4 models), plus a few MB per cached road-signature pack.
- When GNSS is healthy and speed is stable, drop sampling/inference cadence to save battery (the system needs the most inertial evidence during blackout, not during normal GNSS-locked driving). Return to full cadence immediately on GNSS quality drop.

---

## 11. Testing Plan

1. Unit tests per model: input/output shape checks, a fixed-seed forward pass with known expected output range (catch silent breakage from refactors). [Layer 1] For the road-signature classifier specifically, the fixed-seed check should assert on macro recall/F1 too, not just that the output shape is right (see Section 4.7).
2. Fusion core unit tests: feed synthetic sensor data with known ground truth (straight line, constant turn) and check the UKF converges to the expected state. [Layer 2]
3. Integration test: run the benchmark replay tool (Section 9) after every significant change to any model or to the fusion core, track drift percentage over time in a simple log so regressions are caught immediately. [Layer 2]
4. On-device test: measure actual latency and battery draw on at least two real Android devices of different price tiers, not just an emulator. [Layer 3]
5. Field test: at least one real GNSS-blackout test (an actual tunnel or underground parking lot) before the finale, on at least two different vehicle types. [Layer 3, with Layer 1 support for logging/labeling the resulting run]

---

## 12. Task Breakdown (three layers, two people each)

We are not staffing this the way the section headers above imply (one person per phase). Six people, three layers, two people per layer. Each layer below is written as a full brief, specific enough that whoever actually sits down to write the code for a given task, whether that's the two people assigned to the layer working side by side, one of them working solo on their half, or one of them pairing with an AI coding session to move faster through the mechanical parts, can read it and start without needing a live explanation first. Read the full section referenced, not just the one-line deliverable here, the contracts and specs live in the earlier sections.

### Layer 1 - Data & Models (2 people)

Covers Section 3, Section 4 (all six subsections), and the model-level half of Section 11.1. This is the heaviest layer by raw scope (it collapses what would otherwise be four separate roles - data engineer plus three ML engineers - into two people), but every task inside it is a self-contained, well-specified training job against a fixed input/output contract, which makes it the layer easiest to drop into partway through if someone else's time frees up later.

- **Person A**: Build Section 3 end to end first (download IO-VNBD, resampling/windowing/normalization/splitting scripts, produce the tensors everything downstream consumes). Once `data/processed/` exists, move to Section 4.1 (alignment net) and 4.2 (Channel A) - these are the two models everything else is easiest to sanity-check against, since Channel A's target (velocity RMSE under 0.5 m/s) is the most directly comparable to raw GNSS ground truth.
- **Person B**: Once Section 3's tensors exist, build Section 4.3 (Channel B) and 4.4 (road-signature classifier). The road-signature classifier needs real collected corridor data (Section 3.1's "secondary" dataset), so coordinate collection-drive timing with whoever is doing the field/vehicle sessions rather than blocking on it silently.
- **Both, once 4.1-4.4 have first working versions**: Section 4.5 (calibration adapter, lowest priority of the five, has a documented fallback in Section 14 if it runs out of time) and Section 4.6 (export/quantization, applies to all five models, worth doing together once there is more than one model ready so the pipeline is proven once, not five times).
- Priority order if time runs short: Alignment + Channel A first, Channel B second, road-signature third, calibration adapter last.
- Deliverables: `data/processed/` plus preprocessing README; trained weights and quantized ONNX exports for all five models; a metrics report per model against the targets in Section 4 (for the road-signature classifier, that means accuracy, macro recall, and macro F1 per Section 4.7, not accuracy alone); at least one road-signature pack trained on real collected data; a before/after calibration accuracy demo.

### Layer 2 - Fusion & Map (2 people, C++ heavy)

Covers Section 5, Section 6, Section 9, and the fusion/integration half of Section 11. This is the layer that actually does the C++ work (Eigen-based UKF port, the shared `fusion_core` library both other layers link against), and the one layer that does not need to wait on Layer 1 to start real work, see the note at the top of Section 5.5.

- **Person A**: Section 5 - Python prototype of the UKF first (synthetic data, does not need trained models yet), then the Eigen C++ port once the filter logic and tuning are locked. `fusion_core`'s interface is a contract Layer 3 links against directly, do not change its shape without telling them.
- **Person B**: Section 9 - benchmark replay tool, start immediately in parallel with Person A's prototype work using dummy/random values in place of model outputs, swap in real ones as Layer 1 delivers them. Then Section 6 - HMM map matching, once an OSM extract for at least one real corridor is available.
- **Both, once the C++ port is real**: Section 11.2 (fusion unit tests against synthetic ground truth) and 11.3 (integration test via the benchmark tool after every significant change).
- Deliverables: `fusion_core` (Python and C++), `map_matching` tested against a real corridor's OSM extract, benchmark tool showing real drift numbers against IO-VNBD held-out routes, fusion unit tests passing.

### Layer 3 - Platform & Edge (2 people)

Covers Section 7, Section 8, and the on-device/field half of Section 11. Both people here also touch C++ (the NDK bridge, the edge engine), but the focus is integration and platform work rather than the core filter math Layer 2 owns.

- **Person A**: Section 7.1 (raw sensor ingest, native ring buffer so the JVM GC never touches the hot path, FIR pre-filter) and the JNI bridge to `fusion_core`. Once that bridge works, move to Section 8 (edge engine) - it links the same `fusion_core` library with a different sensor ingest/I-O layer, so the skills transfer directly.
- **Person B**: Section 7.2 (UI, Jetpack Compose), 7.3 (runtime loop wiring all the pieces together), 7.4 (calibration-drive user flow).
- **Both**: Section 11.4 (on-device latency/battery test on at least two real devices of different price tiers) and 11.5 (field GNSS-blackout test), with Layer 1 support afterward for logging/labeling whatever raw data comes out of that field run.
- Deliverables: Android app runtime loop working end to end on a real device (rough UI is fine at first), edge-engine binary streaming fused position on reference hardware.

---

## 13. Checkpoints

Checkpoints, not weeks. The ORDER below should not change (later checkpoints genuinely depend on earlier ones being real and tested, not just planned), but how long each one takes is whatever it takes, don't force a calendar onto it.

- **Checkpoint 0 (kickoff, all three layers start the same day)**: Layer 1 downloading IO-VNBD and starting preprocessing scripts. Layer 2 building the UKF process model and its synthetic-data unit tests (this does not need Layer 1's models, see Section 5.5), plus the benchmark tool skeleton running on dummy outputs. Layer 3 scaffolding the Android project and edge-engine project, and getting real raw IMU data flowing into a ring buffer even before there is a model to feed it into.
- **Checkpoint 1**: Layer 1's processed tensors and `norm_stats.json` exist and are documented. Layer 2's Python UKF prototype passes its synthetic-data unit tests. Layer 3's sensor ingest is logging real raw IMU data on a real device.
- **Checkpoint 2**: Layer 1 has Alignment + Channel A trained to a first working version, with the ONNX export/quantization path proven end to end for at least that one model. Layer 2 swaps a first real model output into the benchmark tool in place of a dummy value. Layer 3's JNI bridge successfully calls into `fusion_core`.
- **Checkpoint 3**: Layer 1 has all five models at a first working version (not final accuracy). Layer 2's C++ port of the fusion core is underway.
- **Checkpoint 4**: Layer 2's C++ port is done, map matching is done, the benchmark tool is showing real drift numbers on IO-VNBD held-out routes.
- **Checkpoint 5**: Layer 3's Android runtime loop works end to end on a real device (UI can still be rough). Edge engine bring-up on reference hardware.
- **Checkpoint 6**: Layer 1's calibration adapter works on-device, at least one real vehicle collection session is done, and a road-signature pack is trained on real collected data with accuracy/recall/F1 reported per Section 4.7.
- **Checkpoint 7 (finale prep)**: integration testing across all three layers, at least one real field blackout test, UI polish, screening/demo materials ready.

---

## 14. Known Risks and Fallbacks

- If on-device LoRA fine-tuning (Section 4.5) proves too slow or unstable on real hardware, fall back to a simpler per-vehicle scale/bias correction (just two scalar parameters per channel, fit with a closed-form least-squares solve instead of gradient-based fine-tuning).
- If the road-signature classifier's accuracy is too low on a given corridor (noisy road surface, insufficient training passes), disable the drift-anchor for that corridor rather than letting a wrong reset degrade the trajectory, fall back to Channel A/B fusion alone for that stretch.
- If int8 quantization drops a model's accuracy too much, use float16 quantization instead for that specific model, the latency cost is small relative to the accuracy this recovers.
- If UKF numerical stability is an issue (common with a 7-state filter under fast sigma-point parameter choices), reduce alpha toward 1e-4 and re-tune Q/R before considering a full filter redesign.