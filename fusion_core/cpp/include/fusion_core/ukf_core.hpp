// Generic Unscented Kalman Filter machinery (Van der Merwe scaled
// sigma points), independent of this system's specific state vector
// or measurement sources - those live in ukf.hpp/DualChannelUkf.
//
// Mirrors the algorithm filterpy's UnscentedKalmanFilter implements
// (fusion_core/python_prototype/ukf.py uses that library directly),
// re-derived here in Eigen since there is no equivalent off-the-shelf
// C++ UKF dependency pulled into docker/cpp.Dockerfile. Sign
// convention for sigma points (Cholesky columns vs. filterpy's
// upper-triangular rows) differs cosmetically from filterpy's
// internals but is mathematically equivalent - both satisfy
// S S^T = (n + lambda) * P, which is all the weighted-sum math below
// actually depends on.
#pragma once

#include <functional>

#include <Eigen/Dense>

namespace fusion_core {

// State-transition function: x_k, dt, gyro_yaw -> x_{k+1}.
using FxFn = std::function<Eigen::VectorXd(const Eigen::VectorXd&, double, double)>;
// Measurement function: x -> z (dimension depends on which source is
// being applied this update call).
using HxFn = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

class UnscentedKalmanFilterCore {
 public:
  UnscentedKalmanFilterCore(int n, double alpha, double beta, double kappa);

  void setState(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);

  const Eigen::VectorXd& x() const { return x_; }
  const Eigen::MatrixXd& P() const { return P_; }

  // Predict step: regenerates sigma points from the current x/P,
  // propagates them through `fx`, and forms the prior mean/covariance
  // (with `Q` added). Leaves `sigmasF_` populated for the update(s)
  // that follow this cycle.
  void predict(double dt, double gyroYaw, const Eigen::MatrixXd& Q, const FxFn& fx);

  // Sequential measurement update against one source. Safe to call
  // multiple times per cycle (GNSS, then Channel A, then Channel B,
  // then road-signature anchor) as long as `refreshSigmas()` is
  // called between calls - see that method's docstring for why.
  void update(const Eigen::VectorXd& z, const Eigen::MatrixXd& R, const HxFn& hx);

  // Regenerate `sigmasF_` from the *current* x/P (equivalent to
  // filterpy's `compute_process_sigmas(dt=0, fx=identity)`, per
  // fusion_core/python_prototype/ukf.py's `_refresh_sigmas`
  // docstring). Required between sequential `update()` calls in the
  // same cycle: without it, the second/third/fourth update of a cycle
  // folds its correction into the *pre-update* sigma spread instead
  // of the post-update one, which is mathematically inconsistent and
  // reliably drives P non-positive-definite within a handful of
  // cycles (this was an actual bug hit and fixed in the Python
  // prototype first - see that file's `_refresh_sigmas` docstring for
  // the full story).
  void refreshSigmas();

  // Symmetrize P and add a small diagonal jitter. Cheap insurance
  // against the sigma-point Cholesky step failing on a P that is only
  // asymmetric by floating-point noise, or that has a state variance
  // collapsed near enough to zero (e.g. `ve` on a perfectly straight
  // route) to appear numerically indefinite. Call after every
  // predict() and update() - mirrors the Python prototype's
  // `_symmetrize_p`.
  void stabilizeCovariance(double jitter = 1e-9);

 private:
  Eigen::MatrixXd sigmaPoints(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) const;

  int n_;
  double alpha_;
  double beta_;
  double kappa_;
  double lambda_;
  Eigen::VectorXd wm_;
  Eigen::VectorXd wc_;

  Eigen::VectorXd x_;
  Eigen::MatrixXd P_;
  Eigen::MatrixXd sigmasF_;  // n x (2n+1), sigma points post-fx (or post-refresh)
};

}  // namespace fusion_core
