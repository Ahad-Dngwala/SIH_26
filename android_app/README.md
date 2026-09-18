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
5. Blackout toggle with live drift readout - `MainActivity.kt`'s Compose UI plus
   `SessionRecordingService.ACTION_SET_BLACKOUT`.
6. Three-tab field-test UI - `Live` (status, trajectory canvas, blackout toggle),
   `Sessions` (history with per-session detail, export, delete, and automatic
   recovery of any recording whose sidecar never got written), `Diagnostics` (raw
   front-end status string, IMU rate, filter latency, ZUPT state, Channel P speed,
   and all three track positions unrounded). Diagnostics is deliberately off the
   main screen; see `Master_Implementation_Plan.md`'s "Field-test UI increment".
7. A partial wake lock is held for the duration of a recording
   (`SessionRecordingService`) so screen-off Doze throttling cannot silently reduce
   the sampling rate below what was measured against.
8. **Not yet driven.** No real recording exists; physical-device logging is the
   next acceptance gate.

The Kotlin 2.0 build applies `org.jetbrains.kotlin.plugin.compose`, rather than the
older standalone Compose compiler extension, so its compiler version stays aligned
with Kotlin. **An Android SDK, Java, Gradle wrapper, and Kotlin CLI are not available
in this environment, so the app has not been built with Gradle or run on a device or
emulator here.** The Android-free `fusion/` package is exercised by two correctness
gates that do not need the SDK:

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

- The Gradle/AGP/Kotlin plugin versions in `build.gradle.kts` are current as of this
  writing but were chosen without a live SDK to confirm against; if Android Studio
  reports an incompatible version, take its suggested fix - the specific version
  numbers are not load-bearing, only the min SDK (26) and the fact that `fusion/` has
  no Android imports are.
- The notification's small icon still points at a stock system drawable
  (`android.R.drawable.ic_menu_mylocation`); a status-bar icon needs the same
  alpha-only-drawable constraints a launcher icon does not, and was left alone
  rather than risk a malformed one with no way to render-check it here. The
  launcher icon itself (`@mipmap/ic_launcher`, adaptive, `res/drawable/ic_launcher_*.xml`)
  is a real icon now, not the stock placeholder.

- [`app/`](app/README.md) - the Kotlin/Compose application.
- [`native/`](native/README.md) - historical JNI/NDK notes; not part of the active
  Kotlin-first phone demo.
