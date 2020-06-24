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
						 const cv::Size2i& imageSize,
						 const common::EventSequence& events,
						 const cv::Rect2d& initPatch)
		: normalizedIntegratedNabla_(normalizedIntegratedNabla),
		  patch_(patch),
		  imageSize_(imageSize),
		  initPatch_(initPatch),
		  events_(events)
	{
		gradInterpolator_ = interpolator;
	}

	template <typename T>
	bool operator()(const T* sPose2D, const T* sFlowDir, T* sResiduals) const
	{
		T normPredictedNabla(1e-5);
		warp(sPose2D, sFlowDir, sResiduals, normPredictedNabla);
		cv::Mat motionCompensatedIntegratedNabla =
			cv::Mat::zeros(patch_.height, patch_.width, CV_64F);
		integrateMotionCompensatedEvents(sPose2D,
										 motionCompensatedIntegratedNabla);
		cv::Mat normedNabla = motionCompensatedIntegratedNabla /
							  cv::norm(motionCompensatedIntegratedNabla);
		const auto mean = cv::mean(motionCompensatedIntegratedNabla);
		double var = 0;
		for (int y = 0; y < static_cast<int>(patch_.height); y++)
		{
			for (int x = 0; x < static_cast<int>(patch_.width); x++)
			{
				var += std::pow(
					mean[0] - motionCompensatedIntegratedNabla.at<double>(y, x),
					2);
			}
		}

		var /= (patch_.height * patch_.width);

		for (int y = 0; y < static_cast<int>(patch_.height); y++)
		{
			for (int x = 0; x < static_cast<int>(patch_.width); x++)
			{
				sResiduals[x + y * static_cast<int>(patch_.width)] =
					sResiduals[x + y * static_cast<int>(patch_.width)] /
						ceres::sqrt(normPredictedNabla) +
					T(normedNabla.at<double>(y, x));
			}
		}
		sResiduals[static_cast<int>(patch_.width * patch_.height)] = T(-var);
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

	template <typename T>
	void integrateMotionCompensatedEvents(
		const T* sPose2D, cv::Mat& motionCompensatedIntegratedNabla) const
	{
		Eigen::Map<Sophus::SE2<T> const> pose2D(sPose2D);

		// Simple Motion Compensation formula:
		// event_t = event_0 + (t - t_event) / (t_final - t_init) * dir
		const T dir[2] = {
			pose2D.translation().x() +
				T(patch_.tl().x - initPatch_.tl().x),
			pose2D.translation().y() +
				T(patch_.tl().y - initPatch_.tl().y)};
		const auto t_dif = static_cast<double>(
			(events_.back().timestamp - events_.front().timestamp).count());
		const auto t = static_cast<double>(
			(events_.back().timestamp + events_.front().timestamp).count() *
			0.5);

		for (const auto& event : events_)
		{
			T point[2] = {T(event.value.point.x), T(event.value.point.y)};
			const T compEventX =
				point[0] + (t - static_cast<double>(event.timestamp.count())) /
							   t_dif * dir[0];
			const T compEventY =
				point[1] + (t - static_cast<double>(event.timestamp.count())) /
							   t_dif * dir[1];

			common::Point2i compEvent;
			if constexpr (std::is_same<T, double>::value)
			{
				compEvent = {int(compEventX), int(compEventY)};
			}
			else
			{
				compEvent = {int(compEventX.a), int(compEventY.a)};
			}

			if (patch_.contains(compEvent))
			{
				const auto point = common::Point2i(compEvent.x - patch_.tl().x,
												   compEvent.y - patch_.tl().y);
				motionCompensatedIntegratedNabla.at<double>(point.y, point.x) +=
					static_cast<int32_t>(event.value.sign);
			}
		}
	}

	cv::Mat normalizedIntegratedNabla_;
	Interpolator* gradInterpolator_;
	cv::Rect2d patch_;
	cv::Size2i imageSize_;
	cv::Rect2d initPatch_;
	common::EventSequence events_;
};
}  // namespace tracker