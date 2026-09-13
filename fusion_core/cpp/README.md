# fusion_core/cpp/

[Layer 2 - Fusion & Map] MIP Section 5.5 step 2. C++17/Eigen production
port of `../python_prototype/`, built only once the filter logic and
tuning (Q, R matrices, thresholds) are locked in Python. Becomes the
shared library linked by both `android_app/native/` (JNI bridge) and
`edge_engine/src/`.

No `CMakeLists.txt` yet - add one once there's real source to build.
The `cpp-dev` Docker service (see root `docker-compose.yml`) already
has the toolchain ready: `build-essential`, `cmake`, `libeigen3-dev`,
`libgtest-dev` - `docker compose run --rm cpp-dev bash` and start from
there.

Status: **first working version built** - 1:1 port of
`../python_prototype/ukf.py`'s `DualChannelUkf` (same state vector,
same four sequential update sources, same trust-weighting and GNSS
re-admission ramp), using Eigen directly for the sigma-point UKF
machinery (`ukf_core.{hpp,cpp}` - no off-the-shelf C++ UKF dependency
exists in `docker/cpp.Dockerfile`, so this is hand-rolled Van der
Merwe scaled sigma points, algorithmically equivalent to what filterpy
does for the Python side).

Build and test:

```
docker compose run --rm cpp-dev bash
cmake -S fusion_core/cpp -B fusion_core/cpp/build
cmake --build fusion_core/cpp/build
ctest --test-dir fusion_core/cpp/build --output-on-failure
```

Same 6 unit tests as `../python_prototype/tests/test_ukf.py`
(`tests/test_ukf.cpp`, gtest) - keep both files in sync if either
changes. `tests/parity_demo.cpp` is a one-off (not wired into ctest)
sanity check replicating `tools/benchmark_replay`'s default synthetic
scenario end-to-end: lands at a similar drift magnitude to the Python
prototype's measured 1.49% (different RNG engines mean it won't match
bit-for-bit, and doesn't need to).

Not yet done: not linked into `android_app/native/` or `edge_engine/`
(both still unstarted, blocked on this existing - which it now does);
no `install()`/packaging rules, since nothing downstream consumes this
as an installed library yet; same open items as the Python prototype
(no real IO-VNBD validation, road-signature anchor untested against a
real classifier, Q/R not tuned against real sensor noise) since this
is a faithful port, not new tuning work.

Also ported: the NHC pseudo-measurement (`enable_nhc` in
`FusionConfig`, defaults `false`) and its regression-guard test - see
`../python_prototype/README.md`'s ablation table and root-cause
writeup for why it's implemented but off by default. Keep both
languages' defaults in sync if this changes.
