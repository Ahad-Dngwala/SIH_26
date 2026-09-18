# android_app/app/

The active phone demo is Kotlin + Jetpack Compose, min SDK 26. It is a
measurement rig first: `SessionRecordingService` records raw accelerometer,
gyroscope, magnetometer, and GPS-provider fixes to the JSONL contract consumed by
`tools/phone_replay/`. It runs in a foreground service on a dedicated sensor thread.

`fusion/` contains the Android-free, Kotlin port of the 7-state UKF, mount leveling,
joint forward-axis estimator, physics speed channel (Channel P), GNSS heading
updates, and software-blackout gate. `MainActivity` hosts a three-tab field-test UI
(Live / Sessions / Diagnostics) rather than a single screen - `ui/TrajectoryCanvas.kt`
is the one dependency worth calling out, a plain Compose `Canvas` that plots GNSS
truth, the fused track, and the no-correction coast baseline distinctly, with no map
tiles and no network dependency. The next acceptance test is one real phone
recording replayed through `python -m tools.phone_replay.run`.

ONNX inference, JNI/NDK, Room, MapLibre, calibration UX, and live HMM matching are
deferred; do not add them to the demo path without replacing this decision in the
implementation plan.
