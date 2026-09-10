# android_app/app/

[Layer 3 - Platform & Edge] MIP Section 7.2-7.4. Kotlin, Jetpack
Compose.

- Foreground service with a persistent notification, for continuous
  background sensor logging/inference (Android requires this to keep
  sampling while the screen is off or another app is in front).
- ONNX Runtime Mobile for inference - one `OrtSession` per model, kept
  warm (not recreated per inference call).
- JNI bridge to `../native/` for the UKF step.
- Room database (SQLite) for on-device logging of raw windows used
  later for calibration/retraining - **opt-in only** (see the
  Privacy note in `HLD/main.tex` Section 7.4).
- MapLibre GL (or Mapbox Navigation SDK, offline tile mode) for the
  navigation UI.

**Runtime loop, target <40ms end to end (Section 7.3):** pull latest
window -> alignment net (periodic, not every cycle) -> Channel A +
Channel B in parallel -> compute A/B disagreement, adjust R (Section
5.4) -> road-signature classifier if due, check 0.85 confidence
threshold -> UKF update via JNI -> map-matching -> update UI.

**Calibration flow, user-facing (Section 7.4):** prompt a "calibration
drive" on first use with a new vehicle while GNSS is available -> log
10-30s raw IMU+GNSS -> run the on-device adapter fine-tune (Section
4.5) in the background with a progress indicator -> save the resulting
adapter under a named vehicle profile.

Status: not yet started - no Gradle project exists yet, see
`../README.md`.
