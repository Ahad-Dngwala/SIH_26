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

**Design notes to carry into this work when it starts** (flagged
during a review pass, not yet acted on since nothing here is built
yet): (1) the native ring buffer above must be a pre-allocated
fixed-size buffer (e.g. `std::array` with a rolling write index),
never one that grows via `new`/`push_back` on the 100Hz hot path -
repeated small allocations in a real-time native loop cause heap
fragmentation and unpredictable stalls, defeating the whole point of
avoiding JVM GC here. (2) the JNI bridge should use
`ByteBuffer.allocateDirect()` on the Kotlin side so native code can
read sensor windows via `GetDirectBufferAddress` without a copy each
cycle - copying 2D sensor windows across JNI 100 times/second the
naive way risks eating into the 40ms cycle budget (Section 7.3).
