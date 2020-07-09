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
		sigma_ = 0.5;
	};

	template <typename T>
	bool operator()(const T* motion, T* residual) const
	{
		Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> compensatedEventImage_;
		compensatedEventImage_.setZero(3 * patchRect_.height,
									   3 * patchRect_.width);

		std::cout << "\nPatch " << patchRect_ << std::endl;

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

			for (int i = -2; i <= 2; i++)
			{
				for (int j = -2; j <= 2; j++)
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
						compensatedEventImage_(pointY, pointX) += value;
					}
				}
			}
		}
		// Calculate loss on compensated event image
		//		T mean = compensatedEventImage_.mean();
		T mean = T(0.0);
		int counterNonZero = 1;
		for (int i = 0; i < 3 * patchRect_.height; i++)
		{
			for (int j = 0; j < 3 * patchRect_.width; j++)
			{
				if (compensatedEventImage_(i, j) > T(0.0))
				{
					mean += compensatedEventImage_(i, j);
					counterNonZero++;
				}
			}
		}
		mean = mean / T(counterNonZero);
		if (mean > T(0))
		{
			//			residual[0] = T(1e3) - (compensatedEventImage_ -
			// mean).matrix().norm();
			T std = T(0);
			for (int i = 0; i < 3 * patchRect_.height; i++)
			{
				for (int j = 0; j < 3 * patchRect_.width; j++)
				{
					if (compensatedEventImage_(i, j) > T(0.0))
					{
						std += (compensatedEventImage_(i, j) - mean) *
							   (compensatedEventImage_(i, j) - mean);
					}
				}
			}
			residual[0] = T(1e3) - (std / T(counterNonZero));
		}
		else
		{
			residual[0] = T(1e3) + T(1e3) * (motion[0] * motion[0]) +
						  T(1e3) * (motion[1] + motion[1]);
		}

		std::cout << "Mean: " << mean << std::endl;
		std::cout << "Residual " << residual[0] << std::endl;
		std::cout << "Motion: " << motion[0] << ", " << motion[1] << std::endl;
		return true;
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

	std::list<common::EventSample> events_;
	cv::Rect2i patchRect_;
	common::timestamp_t timestamp_;
	double sigma_;
};
}  // namespace tracker