#pragma once

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace tracker
{
struct contrastFunctor
{
	contrastFunctor(const std::list<common::EventSample>& events,
					const cv::Rect2i patchRect, const double compensateScale)
		: events_(events),
		  patchRect_(patchRect),
		  compensateScale_(compensateScale)
	{
		timestamp_ = common::timestamp_t(static_cast<int32_t>(
			(events.front().timestamp + events.back().timestamp).count() *
			0.5));
	};

	template <typename T>
	bool operator()(const T* motion, T* residual) const
	{
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> compensatedEventImage_;
		compensatedEventImage_.setZero(3 * patchRect_.height,
									   3 * patchRect_.width);

		compensateEvents(motion, compensatedEventImage_);

		// Calculate loss on compensated event image
		//		calculateVarianceLoss(motion, compensatedEventImage_, residual);
		calculateEdgeLoss(motion, compensatedEventImage_, residual);
		return true;
	}

	template <typename T>
	void compensateEvents(const T* motion,
						  Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>&
							  compensatedEventImage) const
	{
		for (const auto& event : events_)
		{
			// Simple Motion Compensation formula:
			// event_t = event_0 + (t - t_event) / (t_final - t_init) * dir
			const T compEventX =
				T(event.value.point.x) +
				T((timestamp_ - event.timestamp).count() * compensateScale_) *
					motion[0];
			const T compEventY =
				T(event.value.point.y) +
				T((timestamp_ - event.timestamp).count() * compensateScale_) *
					motion[1];

			common::Point2i compEvent;
			if constexpr (std::is_same<T, double>::value)
			{
				compEvent = {int(compEventX), int(compEventY)};
			}
			else
			{
				compEvent = {int(compEventX.a), int(compEventY.a)};
			}

			for (int i = -gaussianCompensateKernelSize_;
				 i <= gaussianCompensateKernelSize_; i++)
			{
				for (int j = -gaussianCompensateKernelSize_;
					 j <= gaussianCompensateKernelSize_; j++)
				{
					const int pointX =
						compEvent.x + i - patchRect_.tl().x + patchRect_.width;
					const int pointY =
						compEvent.y + j - patchRect_.tl().y + patchRect_.height;

					if (pointX >= 0 && pointX < 3 * patchRect_.width &&
						pointY >= 0 && pointY < 3 * patchRect_.height)
					{
						const T value =
							gaussian(compEventX, compEventY, T(compEvent.x + i),
									 T(compEvent.y + j), sigmaCompensate_);
						compensatedEventImage(pointY, pointX) += value;
					}
				}
			}
		}
	}

	template <typename T>
	T gaussian(const T meanX, const T meanY, const T x, const T y,
			   const double sigma) const
	{
		T sigmaSq = T(sigma * sigma);
		T normCoef = T(1) / (T(2 * M_PI) * sigmaSq);
		return normCoef *
			   exp(T(-0.5) / sigmaSq *
				   ((x - meanX) * (x - meanX) + (y - meanY) * (y - meanY)));
	}

	template <typename T>
	void calculateVarianceLoss(
		const T* motion,
		const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>&
			compensatedEventImage,
		T* residual) const
	{
		//		T mean = compensatedEventImage_.mean();
		T mean = T(0.0);
		int counterNonZero = 1;
		for (int i = 0; i < 3 * patchRect_.height; i++)
		{
			for (int j = 0; j < 3 * patchRect_.width; j++)
			{
				if (compensatedEventImage(i, j) > T(0.0))
				{
					mean += compensatedEventImage(i, j);
					counterNonZero++;
				}
			}
		}
		mean = mean / T(counterNonZero);

		T std = T(0);
		if (mean > T(0))
		{
			//			residual[0] = T(1e3) - (compensatedEventImage_ -
			// mean).matrix().norm();
			for (int i = 0; i < 3 * patchRect_.height; i++)
			{
				for (int j = 0; j < 3 * patchRect_.width; j++)
				{
					if (compensatedEventImage(i, j) > T(0.0))
					{
						std += (compensatedEventImage(i, j) - mean) *
							   (compensatedEventImage(i, j) - mean);
					}
				}
			}
			std = std / T(counterNonZero);
			residual[0] = T(maxPossibleResidual_) - std;
		}
		else
		{
			// Events went outside of 3 * patch_size
			residual[0] =
				T(maxPossibleResidual_) *
				(T(1) + (motion[0] * motion[0]) + (motion[1] * motion[1]));
		}
	}

	template <typename T>
	void calculateEdgeLoss(
		const T* motion,
		const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>&
			compensatedEventImage,
		T* residual) const
	{
		if (compensatedEventImage.mean() <= T(0.0001))
		{
			// Events went outside of 3 * patch_size. Too much motion!
			residual[0] =
				T(maxPossibleResidual_) *
				(T(1) + (motion[0] * motion[0]) + (motion[1] * motion[1]));
		}
		else
		{
			residual[0] = T(maxPossibleResidual_);
			// Compute gradients
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> gradX;
			gradX.setZero(3 * patchRect_.height, 3 * patchRect_.width);
			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> gradY;
			gradY.setZero(3 * patchRect_.height, 3 * patchRect_.width);
			for (int y = 0; y < 3 * patchRect_.height - 1; y++)
			{
				for (int x = 0; x < 3 * patchRect_.width - 1; x++)
				{
					gradX(y, x) = compensatedEventImage(y, x + 1) -
								  compensatedEventImage(y, x);
					gradY(y, x) = compensatedEventImage(y + 1, x) -
								  compensatedEventImage(y, x);
				}
			}

			// Compute first eigen values of structure tensors at each pixel.
			// Weight with gaussian weighting matrix
			Eigen::Array<T, 2, 2> structureTensor;

			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> firstEigenValues;
			firstEigenValues.setZero(3 * patchRect_.height,
									 3 * patchRect_.width);

			Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> weight;
			weight.setZero(kernelSizeST_ * 2 + 1, kernelSizeST_ * 2 + 1);
			for (int i = -kernelSizeST_; i <= kernelSizeST_; i++)
			{
				for (int j = -kernelSizeST_; j <= kernelSizeST_; j++)
				{
					weight(i + kernelSizeST_, j + kernelSizeST_) =
						gaussian(T(0), T(0), T(j), T(i), sigmaST_);
				}
			}

			for (int y = 0; y < 3 * patchRect_.height - 1; y++)
			{
				for (int x = 0; x < 3 * patchRect_.width - 1; x++)
				{
					structureTensor.setZero();
					for (int i = -kernelSizeST_; i <= kernelSizeST_; i++)
					{
						for (int j = -kernelSizeST_; j <= kernelSizeST_; j++)
						{
							if (x + j >= 0 &&
								x + j < 3 * patchRect_.width - 1 &&
								y + i >= 0 && y + i < 3 * patchRect_.height - 1)
							{
								structureTensor(0, 0) +=
									weight(i + kernelSizeST_,
										   j + kernelSizeST_) *
									gradX(y + i, x + j) * gradX(y + i, x + j);
								structureTensor(0, 1) +=
									weight(i + kernelSizeST_,
										   j + kernelSizeST_) *
									gradX(y + i, x + j) * gradY(y + i, x + j);
								structureTensor(1, 1) +=
									weight(i + kernelSizeST_,
										   j + kernelSizeST_) *
									gradY(y + i, x + j) * gradY(y + i, x + j);
								structureTensor(1, 0) +=
									weight(i + kernelSizeST_,
										   j + kernelSizeST_) *
									gradX(y + i, x + j) * gradY(y + i, x + j);
							}
						}
					}
					T trace = structureTensor(0, 0) + structureTensor(1, 1);
					T det = structureTensor(0, 0) * structureTensor(1, 1) -
							structureTensor(0, 1) * structureTensor(1, 0);
					T diffEigenVals = ceres::sqrt(trace * trace - T(4) * det);
					T firstEigenVal = T(0.5) * (trace + diffEigenVals);

					if (firstEigenVal > T(0))
					{
						firstEigenValues(y, x) = firstEigenVal;
					}
				}
			}

			// Non maximum suppression of first eigen values in blocks
			for (int y = kernelSizeNMS_; y < 3 * patchRect_.height - 1;
				 y += kernelSizeNMS_)
			{
				for (int x = kernelSizeNMS_; x < 3 * patchRect_.width - 1;
					 x += kernelSizeNMS_)
				{
					T maxVal = T(0);
					for (int i = -kernelSizeNMS_; i <= kernelSizeNMS_; i++)
					{
						for (int j = -kernelSizeNMS_; j <= kernelSizeNMS_; j++)
						{
							if (firstEigenValues(y + i, x + j) > maxVal)
							{
								maxVal = firstEigenValues(y + i, x + j);
							}
						}
					}

					if (maxVal > T(0))
					{
						residual[0] -= maxVal / T(maxPossibleResidual_);
					}
				}
			}
		}
	}

	std::list<common::EventSample> events_;
	cv::Rect2i patchRect_;
	common::timestamp_t timestamp_;
	double compensateScale_ = 1e-3;
	double maxPossibleResidual_ = 1e3;

	double sigmaCompensate_ = 1;
	int gaussianCompensateKernelSize_ = 3;

	double sigmaST_ = 1.5;
	int kernelSizeST_ = 3;

	int kernelSizeNMS_ = 1;
};
}  // namespace tracker