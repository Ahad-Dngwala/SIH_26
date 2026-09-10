# Build/test environment for fusion_core/cpp (Section 5.5 step 2) and
# edge_engine/src (Section 8). No CMakeLists.txt exists yet at scaffold
# time - Layer 2/3 add one as they port real code into these folders.
# This image just guarantees everyone has the same toolchain and Eigen
# version instead of debugging "works on my machine" on C++17/Eigen.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libgtest-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["bash"]
