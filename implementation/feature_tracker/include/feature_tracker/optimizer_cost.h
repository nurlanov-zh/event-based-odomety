#pragma once

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace tracker
{
using Grid = ceres::Grid2D<double, 2>;
using GridPtr = std::unique_ptr<Grid>;
using Interpolator = ceres::BiCubicInterpolator<Grid>;
using InterpolatorPtr = std::shared_ptr<Interpolator>;

struct OptimizerCostFunctor
{
	OptimizerCostFunctor(){};

	OptimizerCostFunctor(const cv::Mat& normalizedIntegratedNabla,
						 const std::shared_ptr<Interpolator>& gradInterpolator,
						 const cv::Rect2i& patch)
	{
		normalizedIntegratedNabla_ = normalizedIntegratedNabla;
		interpolator_ = gradInterpolator;
		patch_ = patch;
	}

	template <typename T>
	bool operator()(const T* sPose2D, const T* sFlowDir, T* sResiduals) const
	{
		T normPredictedNabla(1e-5);
		warp(sPose2D, sFlowDir, sResiduals, normPredictedNabla);
		for (int y = 0; y < patch_.height; y++)
		{
			for (int x = 0; x < patch_.width; x++)
			{
				sResiduals[x + y * patch_.width] =
					sResiduals[x + y * patch_.width] /
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

		auto transform = pose2D.matrix2x3();

		const auto center =
			Eigen::Vector2d((patch_.width - 1) / 2, (patch_.height - 1) / 2);

		const Eigen::Matrix<T, 2, 1> offsetToCenter =
			-(pose2D.rotationMatrix() * center) + center;

		for (int y = 0; y < patch_.height; y++)
		{
			for (int x = 0; x < patch_.width; x++)
			{
				T warpedX = transform(0, 0) * T(x) + transform(0, 1) * T(y) +
							transform(0, 2) + offsetToCenter.x();
				T warpedY = transform(1, 0) * T(x) + transform(1, 1) * T(y) +
							transform(1, 2) + offsetToCenter.y();

				// evaluate interpolated gradients at warped points
				T grads[2];
				interpolator_->Evaluate(warpedY, warpedX, grads);

				// compute predicted nabla at this point
				sResiduals[x + patch_.width * y] =
					grads[0] * vx + grads[1] * vy;

				// accumulate norm of predicted nabla
				normPredictedNabla +=
					ceres::pow(sResiduals[x + patch_.width * y], 2);
			}
		}
	}

	cv::Rect2i patch_;
	std::shared_ptr<Interpolator> interpolator_;
	cv::Mat normalizedIntegratedNabla_;
};
}  // namespace tracker