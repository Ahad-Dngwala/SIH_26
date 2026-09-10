# android_app/

[Layer 3 - Platform & Edge] MIP Section 7. Kotlin + Jetpack Compose,
min SDK 26 (Android 8.0). One of the system's two deliverables (with
`edge_engine/`) sharing the same models and `fusion_core` - only the
sensor-ingest and I/O layer differ between the two.

**Not part of this scaffolding pass's code** - only the directory
structure is laid down here. No Gradle project has been generated;
start that from Android Studio's own project wizard rather than
hand-rolling `build.gradle` files here, then commit the result into
`app/`.

**Not containerized.** Android builds need the Android SDK/emulator,
which is a poor fit for a lightweight hackathon Docker setup - use
Android Studio locally. If CI ever needs headless Gradle builds, that's
a separate, later Dockerfile addition, not part of this scaffold.

- [`app/`](app/README.md) - the Kotlin/Compose application (Section 7.2-7.4).
- [`native/`](native/README.md) - NDK/C++ bridge to `fusion_core/cpp/` (Section 7.1, 7.3 step 6).
