#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

static double NormalizeAngle(double rad) {
    while (rad > M_PI) rad -= 2.*M_PI;
    while (rad < -M_PI) rad += 2.*M_PI;
    return rad;
}

static VectorXd Predict(VectorXd state, double delta_t) {
    // extract values for better readability
    double p_x = state(0);
    double p_y = state(1);
    double v = state(2);
    double yaw = state(3);
    double yawd = state(4);
    double nu_a = state(5);
    double nu_yawdd = state(6);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    VectorXd pred(5);
    pred(0) = px_p;
    pred(1) = py_p;
    pred(2) = v_p;
    pred(3) = yaw_p;
    pred(4) = yawd_p;

    return pred;
}


  /**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  n_aug_ = n_x_ + 2;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  Xsig_pred_ = MatrixXd(n_x_, 1 + 2 * n_aug_);
  Xsig_pred_.fill(0.0);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  char* env_std_a = std::getenv("UKF_STD_A");
  if (env_std_a == nullptr) {
    std_a_ = 0.3;
  } else {
    std_a_ = std::stod(env_std_a);
  }

  // Process noise standard deviation yaw acceleration in rad/s^2
  char* env_std_yawdd = std::getenv("UKF_STD_YAWDD");
  if (env_std_yawdd == nullptr) {
    std_yawdd_ = 0.2;
  } else {
    std_yawdd_ = std::stod(env_std_yawdd);
  }
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;

  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_/(lambda_+ n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 weights
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
}

UKF::~UKF() {}

void UKF::CheckNIS() const  {
  std::cout << "std_a_ = " << std_a_ << ", std_yawdd_ = " << std_yawdd_ << std::endl;
  int above_threshold = 0;
  for (const auto nis : nis_radar_) {
    if (nis > 7.815) {
      above_threshold ++;
    }
  }
  std::cout << "Radar NIS > threshold: "
    << static_cast<double>(above_threshold) / static_cast<double>(nis_radar_.size()) << "% ("
    << above_threshold << " times out of " << nis_radar_.size() << ")" << std::endl;

  above_threshold = 0;
  for (const auto nis : nis_laser_) {
    if (nis > 5.991) {
      above_threshold ++;
    }
  }
  std::cout << "Laser NIS > threshold: "
    << static_cast<double>(above_threshold) / static_cast<double>(nis_laser_.size()) << "% ("
    << above_threshold << " times out of " << nis_laser_.size() << ")" << std::endl;
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (not is_initialized_) {
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;

    switch (meas_package.sensor_type_) {
      case MeasurementPackage::LASER:
        x_.head(2) = meas_package.raw_measurements_;
        break;
      case MeasurementPackage::RADAR:
        double r = meas_package.raw_measurements_(0);
        double phi = meas_package.raw_measurements_(1);
        double r_dot = meas_package.raw_measurements_(2);

        x_(0) = r * cos(phi);
        x_(1) = r * sin(phi);
        x_(2) = r_dot;
        break;
    }

    return;
  }

  double dt_s = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt_s);

  switch (meas_package.sensor_type_) {
    case MeasurementPackage::LASER:
      UpdateLidar(meas_package);
      break;
    case MeasurementPackage::RADAR:
      UpdateRadar(meas_package);
      break;
  }

}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  VectorXd x_aug(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_)     = std_a_;
  x_aug(n_x_ + 1) = std_yawdd_;

  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_)               = std_a_;
  P_aug(n_x_ + 1, n_x_ + 1)       = std_yawdd_;

  MatrixXd A = P_aug.llt().matrixL();

  MatrixXd Xsig_aug_(n_aug_, 1 + n_aug_ * 2); 
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug_.col(1 + i)          = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug_.col(1 + i + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    Xsig_pred_.col(i) = Predict(Xsig_aug_.col(i), delta_t);
  }

  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormalizeAngle(x_diff(3));
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z = 2;
  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    // measurement model
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R(n_z, n_z);
  R <<  std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;
  S += R;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  MatrixXd nis = z_diff.transpose() * S.inverse() * z_diff;
  nis_laser_.push_back(nis(0, 0));

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         // r
    Zsig(1, i) = atan2(p_y, p_x);                                     // phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // r_dot
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = NormalizeAngle(z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R(n_z, n_z);
  R <<  std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_ ;
  S += R;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = NormalizeAngle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormalizeAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  z_diff(1) = NormalizeAngle(z_diff(1));

  MatrixXd nis = z_diff.transpose() * S.inverse() * z_diff;
  nis_radar_.push_back(nis(0, 0));

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();
}
