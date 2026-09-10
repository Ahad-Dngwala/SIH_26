# android_app/native/

[Layer 3 - Platform & Edge] MIP Section 7.1, 7.3 step 6. NDK/C++ bridge
between `../app/` and `fusion_core/cpp/`.

- Raw sensor callback via Android's `SensorManager`, `TYPE_ACCELEROMETER`
  / `TYPE_GYROSCOPE` / `TYPE_MAGNETIC_FIELD` at `SENSOR_DELAY_FASTEST`
  or a custom period matching 100Hz. **Do not use
  `TYPE_LINEAR_ACCELERATION`** or any other fused/virtual sensor - it
  applies an undocumented vendor filter that competes with our own
  models (this is a deliberate, explicit design choice, see `HLD/main.tex`
  Section 4 "Why We Avoid a Few Common Choices").
- Sensor callback on a dedicated background thread, never the UI
  thread.
- Native ring buffer so the JVM garbage collector never touches the hot
  path.
- FIR low-pass pre-filter before any window reaches a model (cutoff
  tuned to remove ~20-30Hz road noise while keeping vehicle dynamics -
  confirm the exact cutoff empirically against IO-VNBD).
- JNI bridge calling into `fusion_core/cpp/` for the UKF step.

Status: not yet started - blocked on `fusion_core/cpp/` existing to
bridge to.
