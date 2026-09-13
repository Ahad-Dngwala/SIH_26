// Standalone parity check, not part of the gtest suite: replicates
// tools/benchmark_replay/'s default synthetic_constant_turn scenario
// (config.yaml defaults) using the same formulas as
// route_loader.py/pipeline.py/drift.py, to sanity-check this C++ port
// lands in the same ballpark as the Python prototype's measured
// 1.49% drift (fusion_core/python_prototype/README.md). Exact
// bit-for-bit match isn't expected or attempted - numpy's PCG64 and
// std::mt19937 won't draw the same numbers - this only checks the
// *statistics* line up, not the RNG stream.
//
// Not wired into CMakeLists/ctest on purpose: it's a one-off
// sanity check for this port, not a regression test (test_ukf.cpp
// already covers behavior with proper assertions).
//
// Build: g++ -std=c++17 -O2 -I../include -I/usr/include/eigen3 \
//          parity_demo.cpp ../src/ukf.cpp ../src/ukf_core.cpp -o parity_demo

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "fusion_core/ukf.hpp"

using fusion_core::DualChannelUkf;
using fusion_core::FusionConfig;
using fusion_core::UkfState;

int main() {
  // config.yaml defaults.
  const double durationS = 120.0;
  const double dtS = 0.1;
  const double speedMps = 16.7;
  const double turnRateDps = 3.0;
  const double accelNoiseStd = 0.05;
  const double gyroNoiseStd = 0.01;
  const double blackoutStartS = 40.0;
  const double blackoutEndS = 100.0;
  const double channelNoiseStdMps = 0.3;  // DummyVelocityEstimator default

  const int n = static_cast<int>(std::round(durationS / dtS)) + 1;
  const double turnRateRps = turnRateDps * M_PI / 180.0;

  std::vector<double> t(n), heading(n);
  std::vector<Eigen::Vector2d> vel(n), pos(n);
  for (int i = 0; i < n; ++i) {
    t[i] = i * dtS;
    heading[i] = turnRateRps * t[i];
    vel[i] = Eigen::Vector2d(speedMps * std::cos(heading[i]), speedMps * std::sin(heading[i]));
  }
  pos[0] = Eigen::Vector2d(0.0, 0.0);
  for (int i = 1; i < n; ++i) {
    pos[i] = pos[i - 1] + (vel[i - 1] + vel[i]) / 2.0 * dtS;
  }

  // route_loader.py's gyro_yaw (noisy) - separate RNG streams per
  // signal, mirroring numpy's separate rng.normal(...) calls.
  std::mt19937 gyroRng(0);
  std::normal_distribution<double> gyroNoise(0.0, gyroNoiseStd);
  std::vector<double> gyroYaw(n);
  for (int i = 0; i < n; ++i) gyroYaw[i] = turnRateRps + gyroNoise(gyroRng);
  (void)accelNoiseStd;  // accel_body isn't consumed by this CTCV filter - see ukf.py's fx() note

  std::mt19937 chARng(1);
  std::mt19937 chBRng(2);
  std::normal_distribution<double> channelNoise(0.0, channelNoiseStdMps);

  UkfState initial{pos[0], vel[0], heading[0]};
  DualChannelUkf ukf(initial, FusionConfig());

  std::vector<Eigen::Vector2d> fusedPos(n);
  fusedPos[0] = pos[0];

  int blackoutStartIdx = -1, blackoutEndIdx = -1;
  for (int i = 1; i < n; ++i) {
    const double speedTrue = vel[i].norm();
    const double channelA = speedTrue + channelNoise(chARng);
    const double channelB = speedTrue + channelNoise(chBRng);

    const bool inBlackout = t[i] >= blackoutStartS && t[i] < blackoutEndS;
    if (inBlackout && blackoutStartIdx < 0) blackoutStartIdx = i;
    if (inBlackout) blackoutEndIdx = i;

    // Matches pipeline.py's run_fused_pipeline: GNSS position is
    // exact ground truth (no noise) when available, never a velocity
    // measurement.
    std::optional<Eigen::Vector2d> gnssPos = inBlackout ? std::nullopt : std::optional(pos[i]);

    UkfState state =
        ukf.step(dtS, gyroYaw[i - 1], channelA, channelB, gnssPos, std::nullopt, std::nullopt, 0.0);
    fusedPos[i] = state.pos;
  }

  double errorAtEndM = (fusedPos[blackoutEndIdx] - pos[blackoutEndIdx]).norm();
  double distanceTraveledM = 0.0;
  for (int i = blackoutStartIdx + 1; i <= blackoutEndIdx; ++i) {
    distanceTraveledM += (pos[i] - pos[i - 1]).norm();
  }
  double driftPct = errorAtEndM / distanceTraveledM * 100.0;

  std::cout << "C++ parity demo: drift " << driftPct << "% (" << errorAtEndM << " m over "
            << distanceTraveledM << " m traveled)\n";
  std::cout << "Python prototype's measured drift on this same scenario: 1.49% "
               "(14.87 m over 1000.33 m)\n";
  return 0;
}
