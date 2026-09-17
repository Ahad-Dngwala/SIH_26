# android_app/

[Layer 3 - Platform & Edge] MIP Section 7. Kotlin + Jetpack Compose,
min SDK 26 (Android 8.0). One of the system's two deliverables (with
`edge_engine/`) sharing the same models and `fusion_core` - only the
sensor-ingest and I/O layer differ between the two.

## Status

A Gradle project now exists: `settings.gradle.kts`, root `build.gradle.kts`,
`app/build.gradle.kts`, the manifest, and a working Tier 1 core (items 1-4 and 7 of
`docs/HANDOFF_kotlin_demo_app_v2.md`'s scope ladder):

1. Gradle/Compose project, min SDK 26, foreground service, permissions - done.
2. Sensor ingest at 100 Hz (accel, gyro, magnetometer) plus raw GNSS at 1 Hz -
   `sensors/SessionRecordingService.kt`.
3. Session logger writing the `tools/phone_replay/session.py` schema exactly -
   `sensors/SessionLogger.kt`.
4. UKF port, validated against the parity fixture - `fusion/Ukf.kt`, see
   `tools/parity/README.md`.
5. **Not yet driven.** No real recording exists.
6. Blackout toggle with live drift readout - `MainActivity.kt`'s Compose UI plus
   `SessionRecordingService.ACTION_SET_BLACKOUT`.
7. Minimal UI - present, deliberately plain per the handoff (mode, speed, heading,
   blackout elapsed/distance/drift, IMU rate, step latency).

**This container has no Android SDK, so nothing here has been built with Gradle or
run on a device or emulator.** Every line in `fusion/` has been separately compiled
with plain `kotlinc` (no Android dependencies in that package by design - see
`fusion/LinAlg.kt`'s module docs) and is exercised by two correctness gates that do
not need the SDK:

- `tools/parity/run_kotlin_parity.sh` - the filter math against the Python
  reference's own fixture.
- `tools/phone_replay/check_kotlin_frontend.py` - the front end (mount leveling,
  forward-axis resolution, Channel P, the blackout gate) against the same offline
  chain, on a synthetic session with a known mount rotation.

Both pass as of the last commit to this file. Neither can catch an Android-API
mistake - a wrong permission string, a manifest typo, a `Sensor` API used
incorrectly - because neither compiles against the Android SDK. **The first thing to
do with this code is open it in Android Studio and let Gradle sync**; that will
surface anything wrong at the Android-API layer that `kotlinc` alone cannot see. Two
known gaps to expect on that first sync:

- No launcher icon set exists. The manifest points at a stock system drawable
  (`@android:drawable/ic_menu_mylocation`) specifically so the project resource-links
  without one; replace it via Android Studio's Asset Studio when there is time.
- The Gradle/AGP/Kotlin plugin versions in `build.gradle.kts` are current as of this
  writing but were chosen without a live SDK to confirm against; if Android Studio
  reports an incompatible version, take its suggested fix - the specific version
  numbers are not load-bearing, only the min SDK (26) and the fact that `fusion/` has
  no Android imports are.

- [`app/`](app/README.md) - the Kotlin/Compose application (Section 7.2-7.4).
- [`native/`](native/README.md) - NDK/C++ bridge to `fusion_core/cpp/` (Section 7.1, 7.3 step 6).
