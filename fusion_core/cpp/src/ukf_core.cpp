#include "fusion_core/ukf_core.hpp"

#include <cmath>
#include <stdexcept>

namespace fusion_core {

UnscentedKalmanFilterCore::UnscentedKalmanFilterCore(int n, double alpha, double beta, double kappa)
    : n_(n), alpha_(alpha), beta_(beta), kappa_(kappa) {
  lambda_ = alpha_ * alpha_ * (n_ + kappa_) - n_;

  const int n_sigma = 2 * n_ + 1;
  wm_ = Eigen::VectorXd::Constant(n_sigma, 1.0 / (2.0 * (n_ + lambda_)));
  wc_ = wm_;
  wm_(0) = lambda_ / (n_ + lambda_);
  wc_(0) = wm_(0) + (1.0 - alpha_ * alpha_ + beta_);

  x_ = Eigen::VectorXd::Zero(n_);
  P_ = Eigen::MatrixXd::Identity(n_, n_);
  sigmasF_ = Eigen::MatrixXd::Zero(n_, n_sigma);
}

void UnscentedKalmanFilterCore::setState(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0) {
  x_ = x0;
  P_ = P0;
}

Eigen::MatrixXd UnscentedKalmanFilterCore::sigmaPoints(const Eigen::VectorXd& x,
                                                        const Eigen::MatrixXd& P) const {
  Eigen::LLT<Eigen::MatrixXd> llt((n_ + lambda_) * P);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error(
        "UnscentedKalmanFilterCore::sigmaPoints: Cholesky decomposition failed - "
        "P is not positive definite. Check that stabilizeCovariance() is being "
        "called after every predict()/update().");
  }
  Eigen::MatrixXd L = llt.matrixL();

  Eigen::MatrixXd sigmas(n_, 2 * n_ + 1);
  sigmas.col(0) = x;
  for (int i = 0; i < n_; ++i) {
    sigmas.col(1 + i) = x + L.col(i);
    sigmas.col(1 + n_ + i) = x - L.col(i);
  }
  return sigmas;
}

void UnscentedKalmanFilterCore::predict(double dt, double gyroYaw, const Eigen::MatrixXd& Q,
                                         const FxFn& fx) {
  Eigen::MatrixXd sigmas = sigmaPoints(x_, P_);

  for (int i = 0; i < sigmas.cols(); ++i) {
    sigmasF_.col(i) = fx(sigmas.col(i), dt, gyroYaw);
  }

  x_ = sigmasF_ * wm_;

  Eigen::MatrixXd P_new = Q;
  for (int i = 0; i < sigmasF_.cols(); ++i) {
    Eigen::VectorXd diff = sigmasF_.col(i) - x_;
    P_new += wc_(i) * (diff * diff.transpose());
  }
  P_ = P_new;
}

void UnscentedKalmanFilterCore::update(const Eigen::VectorXd& z, const Eigen::MatrixXd& R,
                                        const HxFn& hx) {
  const int m = static_cast<int>(z.size());
  const int n_sigma = static_cast<int>(sigmasF_.cols());

  Eigen::MatrixXd sigmasH(m, n_sigma);
  for (int i = 0; i < n_sigma; ++i) {
    sigmasH.col(i) = hx(sigmasF_.col(i));
  }

  Eigen::VectorXd zPred = sigmasH * wm_;

  Eigen::MatrixXd Pz = R;
  Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(n_, m);
  for (int i = 0; i < n_sigma; ++i) {
    Eigen::VectorXd dz = sigmasH.col(i) - zPred;
    Pz += wc_(i) * (dz * dz.transpose());
    Eigen::VectorXd dx = sigmasF_.col(i) - x_;
    Pxz += wc_(i) * (dx * dz.transpose());
  }

  Eigen::MatrixXd K = Pxz * Pz.inverse();
  x_ = x_ + K * (z - zPred);
  P_ = P_ - K * Pz * K.transpose();
}

void UnscentedKalmanFilterCore::refreshSigmas() { sigmasF_ = sigmaPoints(x_, P_); }

void UnscentedKalmanFilterCore::stabilizeCovariance(double jitter) {
  Eigen::MatrixXd sym = (P_ + P_.transpose()) / 2.0;
  P_ = sym + Eigen::MatrixXd::Identity(n_, n_) * jitter;
}

}  // namespace fusion_core
