# edge_engine/src/

[Layer 3 - Platform & Edge] MIP Section 8. C++ main loop + sensor
drivers (UART/SPI, ~200Hz FOG-grade IMU). Links the same `fusion_core`
library `android_app/native/` links, with a different ingest/I-O layer
- per `../README.md`, the skills from the Android JNI bridge transfer
directly here (Section 12: "Once that bridge works, move to Section 8 -
it links the same fusion_core library with a different sensor ingest/
I-O layer").

Status: not yet started - blocked on `fusion_core/cpp/` and, ideally,
a working `android_app/native/` bridge to transfer patterns from.
