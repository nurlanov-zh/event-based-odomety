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
		residual[0] = x[0] - y[0];
		residual[1] = x[1] - y[1];
		return true;
	}
};
}  // namespace tracker