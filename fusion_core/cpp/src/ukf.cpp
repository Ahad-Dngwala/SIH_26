#include "fusion_core/ukf.hpp"

#include <algorithm>
#include <cmath>

#include "fusion_core/ukf_core.hpp"

namespace fusion_core {

namespace {

// Process model (Section 5.2): CTCV kinematic propagation. Heading is
// propagated from the (bias-corrected) gyro yaw rate; speed magnitude
// is held constant across the prediction step and only rotated into
// the new heading direction - velocity *changes* come from the
// measurement updates (GNSS, Channel A/B), not from integrating raw
// accelerometer here. See ukf.py's fx() docstring for the full
// rationale (this deliberately avoids double-integrating
// accelerometer through the process model, which is the quadratic-
// drift failure mode HLD/main.tex Section 1 describes).
Eigen::VectorXd fx(const Eigen::VectorXd& x, double dt, double gyroYaw) {
  Eigen::VectorXd xNew = x;

  const double yawRate = gyroYaw - x(BG);
  const double psiNew = x(PSI) + yawRate * dt;

  const double speed = std::hypot(x(VN), x(VE));
  const double vnNew = speed * std::cos(psiNew);
  const double veNew = speed * std::sin(psiNew);

  xNew(PN) = x(PN) + (x(VN) + vnNew) / 2.0 * dt;
  xNew(PE) = x(PE) + (x(VE) + veNew) / 2.0 * dt;
  xNew(VN) = vnNew;
  xNew(VE) = veNew;
  xNew(PSI) = psiNew;
  // ba, bg: constant in the process model (random-walk drift is
  // handled by their entries in Q, not by anything here).
  return xNew;
}

Eigen::VectorXd hxPosition(const Eigen::VectorXd& x) { return Eigen::Vector2d(x(PN), x(PE)); }

Eigen::VectorXd hxPositionVelocity(const Eigen::VectorXd& x) {
  Eigen::VectorXd z(4);
  z << x(PN), x(PE), x(VN), x(VE);
  return z;
}

Eigen::VectorXd hxVelocity(const Eigen::VectorXd& x) { return Eigen::Vector2d(x(VN), x(VE)); }

// See FusionConfig::enable_nhc's docstring (ukf.hpp) - not one of the
// original MIP Section 5.3's four sources, defaults off.
Eigen::VectorXd hxNhc(const Eigen::VectorXd& x) {
  Eigen::VectorXd z(1);
  z(0) = -x(VN) * std::sin(x(PSI)) + x(VE) * std::cos(x(PSI));
  return z;
}

}  // namespace

DualChannelUkf::DualChannelUkf(const UkfState& initialState, const FusionConfig& config)
    : config_(config),
      ukf_(std::make_unique<UnscentedKalmanFilterCore>(kNumStates, config.alpha, config.beta,
                                                         config.kappa)) {
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(kNumStates);
  x0(PN) = initialState.pos(0);
  x0(PE) = initialState.pos(1);
  x0(VN) = initialState.vel(0);
  x0(VE) = initialState.vel(1);
  x0(PSI) = initialState.heading;

  Eigen::VectorXd diag(kNumStates);
  diag << 1.0, 1.0, 0.5, 0.5, 0.05, 0.01, 1e-4;
  ukf_->setState(x0, diag.asDiagonal());
}

DualChannelUkf::~DualChannelUkf() = default;

Eigen::MatrixXd DualChannelUkf::processNoise(double dt, bool blackout) const {
  const double qPos = blackout ? config_.q_pos_blackout : config_.q_pos_gnss;
  const double qVel = blackout ? config_.q_vel_blackout : config_.q_vel_gnss;
  Eigen::VectorXd diag(kNumStates);
  diag << qPos, qPos, qVel, qVel, config_.q_psi, config_.q_ba, config_.q_bg;
  return Eigen::MatrixXd(diag.asDiagonal()) * dt;
}

double DualChannelUkf::channelAR(double channelASpeed, double channelBSpeed) const {
  const double disagreement = std::abs(channelASpeed - channelBSpeed);
  if (disagreement > config_.channel_a_b_disagreement_factor * config_.r_channel_b) {
    return config_.r_channel_a * config_.channel_a_inflated_r_multiplier;
  }
  return config_.r_channel_a;
}

double DualChannelUkf::gnssRMultiplier(double dt, bool gnssAvailable) {
  if (!gnssAvailable) {
    wasBlackedOut_ = true;
    timeSinceGnssReacquiredS_.reset();
    return 1.0;  // unused when GNSS is absent this cycle
  }

  if (wasBlackedOut_) {
    if (!timeSinceGnssReacquiredS_.has_value()) {
      timeSinceGnssReacquiredS_ = 0.0;
    } else {
      *timeSinceGnssReacquiredS_ += dt;
    }

    const double rampFrac =
        std::min(1.0, *timeSinceGnssReacquiredS_ / config_.gnss_reacquire_ramp_s);
    if (rampFrac >= 1.0) {
      wasBlackedOut_ = false;
      return 1.0;
    }
    // Linearly ramp the R multiplier down from "near-zero trust" to
    // nominal (1.0) as rampFrac goes 0 -> 1.
    const double start = config_.gnss_reacquire_r_multiplier_start;
    return start + (1.0 - start) * rampFrac;
  }

  return 1.0;
}

UkfState DualChannelUkf::step(double dt, double gyroYaw, double channelASpeed,
                               double channelBSpeed, std::optional<Eigen::Vector2d> gnssPos,
                               std::optional<Eigen::Vector2d> gnssVel,
                               std::optional<Eigen::Vector2d> roadSignaturePos,
                               double roadSignatureConfidence) {
  const bool blackout = !gnssPos.has_value();

  ukf_->predict(dt, gyroYaw, processNoise(dt, blackout), fx);
  ukf_->stabilizeCovariance();

  if (gnssPos.has_value()) {
    const double rMult = gnssRMultiplier(dt, /*gnssAvailable=*/true);
    if (gnssVel.has_value()) {
      Eigen::VectorXd z(4);
      z << (*gnssPos)(0), (*gnssPos)(1), (*gnssVel)(0), (*gnssVel)(1);
      Eigen::VectorXd rDiag(4);
      rDiag << config_.r_gnss_pos * config_.r_gnss_pos, config_.r_gnss_pos * config_.r_gnss_pos,
          config_.r_gnss_vel * config_.r_gnss_vel, config_.r_gnss_vel * config_.r_gnss_vel;
      Eigen::MatrixXd R = Eigen::MatrixXd(rDiag.asDiagonal()) * (rMult * rMult);
      ukf_->update(z, R, hxPositionVelocity);
    } else {
      Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * config_.r_gnss_pos *
                           config_.r_gnss_pos * (rMult * rMult);
      ukf_->update(*gnssPos, R, hxPosition);
    }
    ukf_->stabilizeCovariance();
    ukf_->refreshSigmas();
  } else {
    gnssRMultiplier(dt, /*gnssAvailable=*/false);
  }

  // Channel A / Channel B: rotate the scalar forward-speed estimates
  // into (vn, ve) using the filter's *current* heading estimate
  // (Section 5.3 point 2/3), then apply as velocity pseudo-
  // measurements.
  const double psiHat = ukf_->x()(PSI);
  const double rChannelA = channelAR(channelASpeed, channelBSpeed);

  Eigen::Vector2d zA(channelASpeed * std::cos(psiHat), channelASpeed * std::sin(psiHat));
  ukf_->update(zA, Eigen::MatrixXd::Identity(2, 2) * rChannelA * rChannelA, hxVelocity);
  ukf_->stabilizeCovariance();
  ukf_->refreshSigmas();

  Eigen::Vector2d zB(channelBSpeed * std::cos(psiHat), channelBSpeed * std::sin(psiHat));
  ukf_->update(zB, Eigen::MatrixXd::Identity(2, 2) * config_.r_channel_b * config_.r_channel_b,
               hxVelocity);
  ukf_->stabilizeCovariance();

  // Non-holonomic constraint - see FusionConfig::enable_nhc's
  // docstring (off by default; measured net-neutral-to-negative on
  // the one benchmark tested, for a specific understood reason).
  if (config_.enable_nhc) {
    ukf_->refreshSigmas();
    Eigen::VectorXd zNhc(1);
    zNhc(0) = 0.0;
    Eigen::MatrixXd rNhc(1, 1);
    rNhc(0, 0) = config_.r_nhc * config_.r_nhc;
    ukf_->update(zNhc, rNhc, hxNhc);
    ukf_->stabilizeCovariance();
  }

  // Road-signature drift-anchor (Section 5.3 point 4 / Section 4.4
  // deployment rule): only applied above the confidence threshold, as
  // a soft position correction, never a hard reset.
  if (roadSignaturePos.has_value() &&
      roadSignatureConfidence >= config_.road_signature_confidence_threshold) {
    ukf_->refreshSigmas();
    Eigen::MatrixXd R =
        Eigen::MatrixXd::Identity(2, 2) * (config_.r_gnss_pos * 2) * (config_.r_gnss_pos * 2);
    ukf_->update(*roadSignaturePos, R, hxPosition);
    ukf_->stabilizeCovariance();
  }

  const Eigen::VectorXd& x = ukf_->x();
  UkfState result;
  result.pos = Eigen::Vector2d(x(PN), x(PE));
  result.vel = Eigen::Vector2d(x(VN), x(VE));
  result.heading = x(PSI);
  return result;
}

Eigen::VectorXd DualChannelUkf::rawState() const { return ukf_->x(); }

}  // namespace fusion_core
