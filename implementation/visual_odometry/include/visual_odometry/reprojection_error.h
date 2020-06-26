#pragma once

#include <common/data_types.h>

namespace visual_odometry
{
struct BundleAdjustmentReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentReprojectionCostFunctor(const Eigen::Vector2d& p2d)
      : p2d(p2d) {}

  template <class T>
  bool operator()(T const* const sTw2c, T const* const sp3dw,
                  T const* const sIntr, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const Tw2c(sTw2c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p3dw(sp3dw);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<CameraModel<T>> cam =
        CameraModel<T>::fromData(sIntr);

    residuals = p2d - cam->project(Tw2c.inverse() * p3dw);
    return true;
  }

  Eigen::Vector2d p2d;
};
} // ns visual_odometry