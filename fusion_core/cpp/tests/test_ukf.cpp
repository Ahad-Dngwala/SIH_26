// MIP Section 11.2: feed synthetic sensor data with known ground
// truth (straight line, constant turn) and check the UKF converges to
// the expected state. Mirrors
// fusion_core/python_prototype/tests/test_ukf.py's four test cases -
// keep the two in sync if either changes.

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "fusion_core/ukf.hpp"

using fusion_core::DualChannelUkf;
using fusion_core::FusionConfig;
using fusion_core::UkfState;

namespace {

struct SyntheticTruth {
  std::vector<double> t;
  std::vector<Eigen::Vector2d> pos;
  std::vector<Eigen::Vector2d> vel;
  std::vector<double> heading;
  std::vector<double> gyroYawTrue;
};

SyntheticTruth generateSyntheticTruth(double durationS, double dtS, double speedMps,
                                       double turnRateDps) {
  const int n = static_cast<int>(std::round(durationS / dtS)) + 1;
  const double turnRateRps = turnRateDps * M_PI / 180.0;

  SyntheticTruth truth;
  truth.t.resize(n);
  truth.pos.resize(n);
  truth.vel.resize(n);
  truth.heading.resize(n);
  truth.gyroYawTrue.resize(n, turnRateRps);

  truth.pos[0] = Eigen::Vector2d(0.0, 0.0);
  for (int i = 0; i < n; ++i) {
    truth.t[i] = i * dtS;
    truth.heading[i] = turnRateRps * truth.t[i];
    truth.vel[i] = Eigen::Vector2d(speedMps * std::cos(truth.heading[i]),
                                    speedMps * std::sin(truth.heading[i]));
  }
  for (int i = 1; i < n; ++i) {
    truth.pos[i] = truth.pos[i - 1] + (truth.vel[i - 1] + truth.vel[i]) / 2.0 * dtS;
  }
  return truth;
}

double pathLength(const std::vector<Eigen::Vector2d>& pos, int start, int end) {
  double total = 0.0;
  for (int i = start + 1; i <= end; ++i) {
    total += (pos[i] - pos[i - 1]).norm();
  }
  return total;
}

}  // namespace

class UkfConvergenceTest : public ::testing::TestWithParam<std::pair<std::string, double>> {};

TEST_P(UkfConvergenceTest, ConvergesWithGnssAvailable) {
  const double turnRateDps = GetParam().second;
  const double durationS = 60.0;
  const double dtS = 0.1;
  const double speedMps = 16.7;

  SyntheticTruth truth = generateSyntheticTruth(durationS, dtS, speedMps, turnRateDps);
  const int n = static_cast<int>(truth.t.size());

  std::mt19937 rng(0);
  std::normal_distribution<double> speedNoise(0.0, 0.05);
  std::normal_distribution<double> gyroNoise(0.0, 0.005);
  std::normal_distribution<double> gnssPosNoise(0.0, 0.5);
  std::normal_distribution<double> gnssVelNoise(0.0, 0.05);

  UkfState initial{truth.pos[0], truth.vel[0], truth.heading[0]};
  DualChannelUkf ukf(initial, FusionConfig());

  std::vector<Eigen::Vector2d> estPos(n);
  estPos[0] = truth.pos[0];

  for (int i = 1; i < n; ++i) {
    const double speedTrue = truth.vel[i].norm();
    const double channelA = speedTrue + speedNoise(rng);
    const double channelB = speedTrue + speedNoise(rng) * 3.0;
    const double gyroMeas = truth.gyroYawTrue[i - 1] + gyroNoise(rng);

    Eigen::Vector2d gnssPos =
        truth.pos[i] + Eigen::Vector2d(gnssPosNoise(rng), gnssPosNoise(rng));
    Eigen::Vector2d gnssVel =
        truth.vel[i] + Eigen::Vector2d(gnssVelNoise(rng), gnssVelNoise(rng));

    UkfState state = ukf.step(dtS, gyroMeas, channelA, channelB, gnssPos, gnssVel, std::nullopt, 0.0);
    estPos[i] = state.pos;
  }

  const int settleIdx = static_cast<int>(2.0 / dtS);
  double sumErr = 0.0;
  for (int i = settleIdx; i < n; ++i) {
    sumErr += (estPos[i] - truth.pos[i]).norm();
  }
  const double meanErr = sumErr / (n - settleIdx);
  EXPECT_LT(meanErr, 3.0) << "mean position error too high with GNSS available: " << meanErr << " m";
}

TEST_P(UkfConvergenceTest, BoundsDriftDuringBlackout) {
  const double turnRateDps = GetParam().second;
  const double durationS = 120.0;
  const double dtS = 0.1;
  const double speedMps = 16.7;
  const double blackoutStartS = 40.0;
  const double blackoutEndS = 100.0;

  SyntheticTruth truth = generateSyntheticTruth(durationS, dtS, speedMps, turnRateDps);
  const int n = static_cast<int>(truth.t.size());

  std::mt19937 rng(1);
  std::normal_distribution<double> aNoise(0.0, 0.3);
  std::normal_distribution<double> bNoise(0.0, 1.0);
  std::normal_distribution<double> gyroNoise(0.0, 0.01);
  std::normal_distribution<double> gnssPosNoise(0.0, 0.5);
  std::normal_distribution<double> gnssVelNoise(0.0, 0.05);

  UkfState initial{truth.pos[0], truth.vel[0], truth.heading[0]};
  DualChannelUkf ukf(initial, FusionConfig());

  std::vector<Eigen::Vector2d> estPos(n);
  estPos[0] = truth.pos[0];

  int endIdx = 0;
  for (int i = 1; i < n; ++i) {
    const double speedTrue = truth.vel[i].norm();
    const double channelA = speedTrue + aNoise(rng);
    const double channelB = speedTrue + bNoise(rng);
    const double gyroMeas = truth.gyroYawTrue[i - 1] + gyroNoise(rng);

    const bool inBlackout = truth.t[i] >= blackoutStartS && truth.t[i] < blackoutEndS;

    std::optional<Eigen::Vector2d> gnssPos;
    std::optional<Eigen::Vector2d> gnssVel;
    if (!inBlackout) {
      gnssPos = truth.pos[i] + Eigen::Vector2d(gnssPosNoise(rng), gnssPosNoise(rng));
      gnssVel = truth.vel[i] + Eigen::Vector2d(gnssVelNoise(rng), gnssVelNoise(rng));
    } else {
      endIdx = i;
    }

    UkfState state = ukf.step(dtS, gyroMeas, channelA, channelB, gnssPos, gnssVel, std::nullopt, 0.0);
    estPos[i] = state.pos;
  }

  const double errorAtEndM = (estPos[endIdx] - truth.pos[endIdx]).norm();
  int startIdx = 0;
  for (int i = 0; i < n; ++i) {
    if (truth.t[i] >= blackoutStartS) {
      startIdx = i;
      break;
    }
  }
  const double distanceTraveledM = pathLength(truth.pos, startIdx, endIdx);
  const double driftPct = errorAtEndM / distanceTraveledM * 100.0;

  // Loose bound mirroring the Python test - not asserting the <10%
  // product benchmark itself here (needs real Channel A/B and the
  // road-signature anchor), just that a correctly-ported UKF
  // meaningfully beats raw open-loop drift.
  EXPECT_LT(driftPct, 15.0) << "drift too high during blackout: " << driftPct << "%";
}

INSTANTIATE_TEST_SUITE_P(StraightAndConstantTurn, UkfConvergenceTest,
                          ::testing::Values(std::make_pair(std::string("straight"), 0.0),
                                             std::make_pair(std::string("constant_turn"), 3.0)));

TEST(UkfRoadSignatureTest, AnchorPullsPositionTowardSegment) {
  UkfState initial{Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(16.7, 0.0), 0.0};
  DualChannelUkf ukf(initial, FusionConfig());

  // Run a few blackout steps with no anchor, drifting slightly.
  for (int i = 0; i < 20; ++i) {
    ukf.step(0.1, 0.02, 16.7, 16.7, std::nullopt, std::nullopt, std::nullopt, 0.0);
  }

  Eigen::Vector2d posBefore = ukf.rawState().head<2>();
  Eigen::Vector2d anchor = posBefore + Eigen::Vector2d(5.0, -5.0);

  UkfState state = ukf.step(0.1, 0.02, 16.7, 16.7, std::nullopt, std::nullopt, anchor, 0.9);

  const double distBefore = (posBefore - anchor).norm();
  const double distAfter = (state.pos - anchor).norm();
  EXPECT_LT(distAfter, distBefore) << "road-signature anchor did not pull position toward the segment";
}

TEST(UkfTrustWeightingTest, ChannelADownweightedOnDisagreement) {
  UkfState initial{Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(16.7, 0.0), 0.0};
  DualChannelUkf ukf(initial, FusionConfig());

  // Channel A reports a wild, wrong speed (simulating a pothole
  // gyro-glitch/mount-slip event); Channel B stays near truth.
  UkfState state = ukf.step(0.1, 0.0, 40.0, 16.7, std::nullopt, std::nullopt, std::nullopt, 0.0);
  const double fusedSpeed = state.vel.norm();
  EXPECT_LT(std::abs(fusedSpeed - 16.7), std::abs(fusedSpeed - 40.0));
}

TEST(UkfNhcTest, DisabledByDefault) {
  // Regression guard for a real finding (see FusionConfig::enable_nhc's
  // docstring, ukf.hpp): enabling NHC measurably hurts drift on this
  // system's synthetic constant-turn benchmark, because Channel A/B's
  // own MIP-specified design already bakes in the same
  // zero-lateral-velocity assumption NHC asserts a second time. If
  // this ever fails because someone flipped the default to true,
  // that's a deliberate design change - it should come with a fresh
  // ablation re-run and a docstring update, not a silent flip.
  EXPECT_FALSE(FusionConfig().enable_nhc);
}
