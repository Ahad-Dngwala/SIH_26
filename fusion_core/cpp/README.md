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

Status: not yet started - blocked on the Python prototype being locked
first (Section 5.5 ordering).
