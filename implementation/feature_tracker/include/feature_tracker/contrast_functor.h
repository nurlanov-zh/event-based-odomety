#pragma once

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace tracker
{
struct contrastFunctor
{
	contrastFunctor(const std::list<common::EventSample>& events,
					const cv::Rect2i patchRect)
		: events_(events), patchRect_(patchRect)
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
				T((timestamp_ - event.timestamp).count() * 1e-3) * motion[0];
			const T compEventY =
				T(event.value.point.y) +
				T((timestamp_ - event.timestamp).count() * 1e-3) * motion[1];

			common::Point2i compEvent;
			if constexpr (std::is_same<T, double>::value)
			{
				compEvent = {int(compEventX), int(compEventY)};
			}
			else
			{
				compEvent = {int(compEventX.a), int(compEventY.a)};
			}

			for (int i = -gaussianBlockSize_; i <= gaussianBlockSize_; i++)
			{
				for (int j = -gaussianBlockSize_; j <= gaussianBlockSize_; j++)
				{
					const int pointX =
						compEvent.x + i - patchRect_.tl().x + patchRect_.width;
					const int pointY =
						compEvent.y + j - patchRect_.tl().y + patchRect_.height;

					if (pointX >= 0 && pointX < 3 * patchRect_.width &&
						pointY >= 0 && pointY < 3 * patchRect_.height)
					{
						const T value =
							gaussian(compEventX, compEventY, compEvent.x + i,
									 compEvent.y + j);
						compensatedEventImage(pointY, pointX) += value;
					}
				}
			}
		}
	}

	template <typename T>
	T gaussian(const T& compEventX, const T& compEventY, const int x,
			   const int y) const
	{
		T sigmaSq = T(sigma_ * sigma_);
		T normCoef = T(1) / (T(2 * M_PI) * sigmaSq);
		return normCoef * exp(T(-0.5) / sigmaSq *
							  ((T(x) - compEventX) * (T(x) - compEventX) +
							   (T(y) - compEventY) * (T(y) - compEventY)));
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
			residual[0] = T(1e3) - std;
		}
		else
		{
			// Events went outside of 3 * patch_size
			residual[0] = T(1e3) * (T(1) + (motion[0] * motion[0]) +
									(motion[1] * motion[1]));
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
			// Events went outside of 3 * patch_size
			residual[0] = T(1e3) * (T(1) + (motion[0] * motion[0]) +
									(motion[1] * motion[1]));
		}
		else
		{
			residual[0] = T(1e3);
			// compute structure tensor
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

			Eigen::Array<T, 2, 2> structureTensor;

			for (int y = 0; y < 3 * patchRect_.height - 1; y++)
			{
				for (int x = 0; x < 3 * patchRect_.width - 1; x++)
				{
					structureTensor.setZero();
					for (int i = -(blockSize_ - 1) / 2;
						 i <= int((blockSize_ - 1) / 2); i++)
					{
						for (int j = -(blockSize_ - 1) / 2;
							 j <= int((blockSize_ - 1) / 2); j++)
						{
							if (x + j >= 0 &&
								x + j < 3 * patchRect_.width - 1 &&
								y + i >= 0 && y + i < 3 * patchRect_.height - 1)
							{
								structureTensor(0, 0) +=
									gradX(y + i, x + j) * gradX(y + i, x + j);
								structureTensor(0, 1) +=
									gradX(y + i, x + j) * gradY(y + i, x + j);
								structureTensor(1, 1) +=
									gradY(y + i, x + j) * gradY(y + i, x + j);
								structureTensor(1, 0) +=
									gradX(y + i, x + j) * gradY(y + i, x + j);
							}
						}
					}
					T trace = structureTensor(0, 0) + structureTensor(1, 1);
					T det = structureTensor(0, 0) * structureTensor(1, 1) -
							structureTensor(0, 1) * structureTensor(1, 0);
					T eigenDiff = ceres::sqrt(trace * trace - T(4) * det);
					T eigenFirstVal = T(0.5) * (trace + eigenDiff);

					if (eigenFirstVal > T(3))
					{
						residual[0] -= eigenFirstVal / T(1e3);
					}
				}
			}
		}
	}

	std::list<common::EventSample> events_;
	cv::Rect2i patchRect_;
	common::timestamp_t timestamp_;
	double sigma_ = 0.5;
	int blockSize_ = 7;
	int gaussianBlockSize_ = 2;
};
}  // namespace tracker