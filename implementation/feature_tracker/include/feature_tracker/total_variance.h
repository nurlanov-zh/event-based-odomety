#pragma once

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace tracker
{
struct totalVarianceFunctor
{
	totalVarianceFunctor() { weight_ = 1.0; };
	totalVarianceFunctor(double weight) : weight_(weight){};
	template <typename T>
	bool operator()(const T* x, const T* y, T* residual) const
	{
		residual[0] = T(weight_) * (x[0] - y[0]);
		residual[1] = T(weight_) * (x[1] - y[1]);
		return true;
	}
	double weight_;
};
}  // namespace tracker