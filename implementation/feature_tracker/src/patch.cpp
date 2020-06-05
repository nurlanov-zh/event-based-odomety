#include "feature_tracker/patch.h"

#include <opencv2/core/eigen.hpp>

namespace tracker
{
Patch::Patch()
{
	init();
}

Patch::Patch(const cv::Rect2i& rect) : patch_(rect)
{
	init();
}

Patch::Patch(const Corner& corner, int extent)
{
	patch_ = cv::Rect2i(corner.x - extent, corner.y - extent, 2 * extent + 1,
						2 * extent + 1);

	init();
}

void Patch::init()
{
	// Just for test visualization
	for (size_t i = 0; i < 30; ++i)
	{
		common::Pose2d pose;
		pose.translation() = Eigen::Vector2d(i + toCorner().x, toCorner().y);
		trajectory_.push_back({pose, common::timestamp_t(0)});
	}

	lost_			 = false;
	numOfEvents_	 = 50;
	integratedNabla_ = cv::Mat::zeros(patch_.height, patch_.width, CV_64F);
	predictedNabla_  = cv::Mat::zeros(patch_.height, patch_.width, CV_64F);
	gradX_			 = cv::Mat::zeros(patch_.height, patch_.width, CV_64F);
	gradY_			 = cv::Mat::zeros(patch_.height, patch_.width, CV_64F);
}

void Patch::addEvent(const common::EventSample& event)
{
	events_.emplace_back(event);
}

void Patch::integrateEvents()
{
	integratedNabla_ = cv::Mat::zeros(patch_.height, patch_.width, CV_64F);
	for (const auto& event : events_)
	{
		if (patch_.contains(event.value.point))
		{
			const auto& point = frameToPatchCoords(event.value.point);
			integratedNabla_.at<double>(point.y, point.x) +=
				static_cast<int32_t>(event.value.sign);
		}
	}
}

void Patch::warpImage()
{
	cv::Mat warpedGradX;
	cv::Mat warpedGradY;

	cv::Mat warpCv;
	cv::eigen2cv(warp_.matrix2x3(), warpCv);

	const auto center =
		Eigen::Vector2d((patch_.width - 1) / 2, (patch_.height - 1) / 2);

	const Eigen::Vector2d offsetToCenter =
		-(warp_.rotationMatrix() * center) + center;
	warpCv.at<double>(0, 2) += offsetToCenter.x();
	warpCv.at<double>(1, 2) += offsetToCenter.y();

	cv::warpAffine(gradX_, warpedGradX, warpCv, {patch_.width, patch_.height});
	cv::warpAffine(gradY_, warpedGradY, warpCv, {patch_.width, patch_.height});

	predictedNabla_ =
		warpedGradX * std::cos(flowDir_) + warpedGradY * std::sin(flowDir_);
}

void Patch::resetBatch()
{
	events_.clear();
}

Corner Patch::toCorner() const
{
	return (patch_.br() + patch_.tl()) * 0.5;
}

bool Patch::isInPatch(const common::Point2i& point) const
{
	return patch_.contains(point);
}

common::Point2i Patch::patchToFrameCoords(
	const common::Point2i& pointInPatch) const
{
	return pointInPatch + patch_.tl();
}

common::Point2i Patch::frameToPatchCoords(
	const common::Point2i& pointInFrame) const
{
	return pointInFrame - patch_.tl();
}

common::EventSequence const& Patch::getEvents() const
{
	return events_;
}

}  // namespace tracker