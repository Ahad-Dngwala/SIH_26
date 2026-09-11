# fusion_core/

[Layer 2 - Fusion & Map] MIP Section 5. The Unscented Kalman Filter
that is the core of the whole system - the one non-ML subsystem
everything else feeds into. Prototyped in Python, then ported to C++
for production; the shared library both `android_app/native/` and
`edge_engine/src/` link against.

**Python prototype has a first working version** - see
`python_prototype/README.md` for what's implemented, tested, and
still open. C++ port (`cpp/`) not started yet - blocked on the Python
filter logic/tuning being locked first, per Section 5.5's own
ordering.

**State vector (Section 5.1):** 7 states `[pn, pe, vn, ve, psi, ba,
bg]` - position north/east (m), velocity north/east (m/s), heading
(rad), residual accelerometer bias, residual gyro bias.

**Does not need Layer 1 to start** (Section 5.5): the process model,
sigma-point math, and 4-source update logic can be built and unit-
tested against synthetic sensor data before any real trained model
exists - this is why Checkpoint 0 has Layer 2 starting the UKF process
model and its synthetic-data unit tests on day one, in parallel with
Layer 1's data pipeline.

**Interface contract Layer 3 depends on:** `android_app/native/` links
against `fusion_core/cpp/`'s headers directly. Don't change that
library's shape without telling Layer 3 first (Section 12).

**Consumed by `tools/benchmark_replay/`** (Section 9) - the Python
prototype here is what the benchmark tool actually runs.
