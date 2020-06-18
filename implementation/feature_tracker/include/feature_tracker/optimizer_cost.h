#pragma once

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace tracker
{
using Grid = ceres::Grid2D<double, 2>;
using GridPtr = std::unique_ptr<Grid>;
using Interpolator = ceres::BiCubicInterpolator<Grid>;
using InterpolatorPtr = std::unique_ptr<Interpolator>;

struct OptimizerCostFunctor
{
	OptimizerCostFunctor(){};

	OptimizerCostFunctor(const cv::Mat normalizedIntegratedNabla,
						 Interpolator* interpolator, const cv::Rect2d& patch,
						 const cv::Size2i& imageSize)
		: normalizedIntegratedNabla_(normalizedIntegratedNabla),
		  patch_(patch),
		  imageSize_(imageSize)
	{
		gradInterpolator_ = interpolator;
	}

	template <typename T>
	bool operator()(const T* sPose2D, const T* sFlowDir, T* sResiduals) const
	{
		T normPredictedNabla(1e-5);
		warp(sPose2D, sFlowDir, sResiduals, normPredictedNabla);
		for (int y = 0; y < static_cast<int>(patch_.height); y++)
		{
			for (int x = 0; x < static_cast<int>(patch_.width); x++)
			{
				sResiduals[x + y * static_cast<int>(patch_.width)] =
					sResiduals[x + y * static_cast<int>(patch_.width)] /
						ceres::sqrt(normPredictedNabla) +
					T(normalizedIntegratedNabla_.at<double>(y, x));
			}
		}
		return true;
	}

	template <typename T>
	void warp(const T* sPose2D, const T* sFlowDir, T* sResiduals,
			  T& normPredictedNabla) const
	{
		Eigen::Map<Sophus::SE2<T> const> pose2D(sPose2D);
		T vx = ceres::cos(sFlowDir[0]);
		T vy = ceres::sin(sFlowDir[0]);

		const auto transform = pose2D.matrix2x3();

		for (int y = 0; y < static_cast<int>(patch_.height); y++)
		{
			for (int x = 0; x < static_cast<int>(patch_.width); x++)
			{
				T warpedX = transform(0, 0) * T(x + patch_.tl().x) +
							transform(0, 1) * T(y + patch_.tl().y) +
							transform(0, 2);
				T warpedY = transform(1, 0) * T(x + patch_.tl().x) +
							transform(1, 1) * T(y + patch_.tl().y) +
							transform(1, 2);

				// evaluate interpolated gradients at warped points
				if (warpedX >= T(imageSize_.width) or
					warpedY >= T(imageSize_.height) or warpedX < T(0.0) or
					warpedY < T(0.0))
				{
					sResiduals[x + static_cast<int>(patch_.width) * y] = T(0.0);
				}
				else
				{
					T grads[2];
					gradInterpolator_->Evaluate(warpedY, warpedX, grads);

					// compute predicted nabla at this point
					sResiduals[x + static_cast<int>(patch_.width) * y] =
						grads[0] * vx + grads[1] * vy;

					// accumulate norm of predicted nabla
					normPredictedNabla += ceres::pow(
						sResiduals[x + static_cast<int>(patch_.width) * y], 2);
				}
			}
		}
	}

	cv::Mat normalizedIntegratedNabla_;
	Interpolator* gradInterpolator_;
	cv::Rect2d patch_;
	cv::Size2i imageSize_;
};
}  // namespace tracker