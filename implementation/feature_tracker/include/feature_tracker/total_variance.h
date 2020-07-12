#pragma once

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace tracker
{
struct totalVarianceFunctor
{
	totalVarianceFunctor(){};
	totalVarianceFunctor(double weight) : weight_(weight){};
	totalVarianceFunctor(double weight, int numParam)
		: weight_(weight), numParam_(numParam){};
	template <typename T>
	bool operator()(const T* x, const T* y, T* residual) const
	{
		for (int i = 0; i < numParam_; i++)
		{
			residual[i] = T(weight_) * ceres::abs((x[i] - y[i]));
		}
		return true;
	}
	double weight_ = 1.0;
	int numParam_ = 2;
};
}  // namespace tracker