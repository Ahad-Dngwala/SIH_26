// C++17/Eigen port of fusion_core/python_prototype/ukf.py (MIP
// Section 5.5 step 2). Same state vector, same four measurement
// sources, same trust-weighting and GNSS re-admission ramp logic -
// port this class 1:1 with the Python prototype rather than
// redesigning, per Section 5.5's own ordering ("once the filter logic
// and tuning are locked [in Python], port to C++"). If you change
// something here, change it in the Python file too (or vice versa)
// so the two don't silently diverge - see fusion_core/README.md's
// interface-contract note: this library is what
// android_app/native/ and edge_engine/src/ both link against.
//
// State vector (Section 5.1), 7 states: [pn, pe, vn, ve, psi, ba, bg]
//   pn, pe - position north/east, meters, local tangent frame
//   vn, ve - velocity north/east, m/s
//   psi    - heading, radians
//   ba     - residual accelerometer bias (reserved, see ukf.py's note)
//   bg     - residual gyro bias, rad/s
#pragma once

#include <memory>
#include <optional>

#include <Eigen/Dense>

namespace fusion_core {

class UnscentedKalmanFilterCore;

constexpr int kNumStates = 7;
enum StateIndex { PN = 0, PE = 1, VN = 2, VE = 3, PSI = 4, BA = 5, BG = 6 };

struct UkfState {
  Eigen::Vector2d pos;  // [north, east], meters
  Eigen::Vector2d vel;  // [v_north, v_east], m/s
  double heading;       // rad
};

// Tunable parameters - defaults match
// fusion_core/python_prototype/ukf.py's FusionConfig exactly. Keep
// the two in sync.
struct FusionConfig {
  double alpha = 1e-3;
  double beta = 2.0;
  double kappa = 0.0;

  // Process noise (Section 5.2): GNSS-available vs. blackout regimes.
  double q_pos_gnss = 0.05;
  double q_vel_gnss = 0.1;
  double q_pos_blackout = 0.5;
  double q_vel_blackout = 0.2;
  double q_psi = 1e-3;
  double q_ba = 1e-5;
  double q_bg = 1e-6;

  // Measurement noise, nominal (Section 5.3).
  double r_gnss_pos = 3.0;      // meters, 1-sigma
  double r_gnss_vel = 0.3;      // m/s, 1-sigma
  double r_channel_a = 0.5;     // m/s, 1-sigma - Channel A target RMSE (Section 4.2)
  double r_channel_b = 1.75;    // m/s, 1-sigma - mid Section 4.3's 1.5-2.0 m/s target band

  // Trust weighting (Section 5.4).
  double channel_a_b_disagreement_factor = 1.5;
  double channel_a_inflated_r_multiplier = 5.0;

  // Road-signature anchor (Section 5.3 point 4 / Section 4.4).
  double road_signature_confidence_threshold = 0.85;

  // GNSS re-admission ramp (Section 5.5 step 5).
  double gnss_reacquire_ramp_s = 2.5;
  double gnss_reacquire_r_multiplier_start = 50.0;
};

// Wraps UnscentedKalmanFilterCore with the CTCV process model and the
// four sequential update sources described in Section 5.3. One
// instance per vehicle/run; call step() once per cycle.
class DualChannelUkf {
 public:
  DualChannelUkf(const UkfState& initialState, const FusionConfig& config = FusionConfig());
  ~DualChannelUkf();

  DualChannelUkf(const DualChannelUkf&) = delete;
  DualChannelUkf& operator=(const DualChannelUkf&) = delete;

  // Advance the filter by one cycle. `gnssPos`/`gnssVel` are
  // std::nullopt when GNSS is unavailable this cycle (blackout).
  // `roadSignaturePos` is std::nullopt when no segment match is
  // offered this cycle; `roadSignatureConfidence` is only consulted
  // when `roadSignaturePos` is set.
  UkfState step(double dt, double gyroYaw, double channelASpeed, double channelBSpeed,
                std::optional<Eigen::Vector2d> gnssPos, std::optional<Eigen::Vector2d> gnssVel,
                std::optional<Eigen::Vector2d> roadSignaturePos, double roadSignatureConfidence);

  // Exposed for unit tests that want to inspect filter internals
  // (e.g. current position estimate) without a full step().
  Eigen::VectorXd rawState() const;

 private:
  Eigen::MatrixXd processNoise(double dt, bool blackout) const;
  double channelAR(double channelASpeed, double channelBSpeed) const;
  double gnssRMultiplier(double dt, bool gnssAvailable);

  FusionConfig config_;
  std::unique_ptr<UnscentedKalmanFilterCore> ukf_;

  bool wasBlackedOut_ = false;
  std::optional<double> timeSinceGnssReacquiredS_;
};

}  // namespace fusion_core
