#pragma once

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace tracker
{
struct totalVarianceFunctor
{
	totalVarianceFunctor(){};
	template <typename T>
	bool operator()(const T* x, const T* y, T* residual) const
	{
		Eigen::Map<Eigen::Matrix<T, 2, 1> const> xV(x);
		Eigen::Map<Eigen::Matrix<T, 2, 1> const> yV(y);
		residual[0] = xV.x() - yV.x();
		residual[1] = xV.y() - yV.y();
		return true;
	}
};
}  // namespace tracker