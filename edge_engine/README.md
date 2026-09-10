# edge_engine/

[Layer 3 - Platform & Edge] MIP Section 8. The system's second
deliverable, alongside `android_app/` - same `fusion_core` C++ library,
same ONNX models, different sensor-ingest and I/O layer. Reference
hardware: Raspberry Pi CM4 or Jetson Orin Nano class board.

**Not part of this scaffolding pass's code** - only the directory
structure is laid down here.

- Sensor ingest: UART or SPI driver reading a FOG-grade IMU at ~200Hz,
  normalized into the same windowed-tensor format the phone app uses
  before touching any model (this shared normalization step is what
  keeps the two deliverables one codebase).
- Output: no navigation UI - streams the fused position (and optionally
  the matched edge ID) as JSON or a simple binary struct over a
  serial/socket interface, for a host system (robotics stack, fleet
  computer) to consume.
- Runtime: ONNX Runtime's C++ API for inference.

The `cpp-dev` Docker service (root `docker-compose.yml`) has the same
C++17/CMake/Eigen toolchain this needs.

- [`src/`](src/README.md) - C++ main loop, sensor drivers.
