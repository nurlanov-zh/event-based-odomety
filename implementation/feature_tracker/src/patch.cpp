#include "feature_tracker/patch.h"
#include "feature_tracker/optimizer.h"

#include <opencv2/core/eigen.hpp>

namespace tracker
{
Patch::Patch(const Corner& corner, int extent,
			 const common::timestamp_t& timestamp)
	: currentTimestamp_(timestamp)
{
	patch_ = cv::Rect2d(corner.x - extent, corner.y - extent, 2 * extent + 1,
						2 * extent + 1);
	init();
}

void Patch::init()
{
	init_ = false;
	lost_ = false;
	trackId_ = -1;
	numOfEvents_ = 75;
	initPoint_ = toCorner();
	integratedNabla_ = cv::Mat::zeros(patch_.height, patch_.width, CV_64F);
	predictedNabla_ = cv::Mat::zeros(patch_.height, patch_.width, CV_64F);
	costMap_ = cv::Mat::zeros(1, 1, CV_64F);

	addTrajectoryPosition();
}

void Patch::addEvent(const common::EventSample& event)
{
	events_.emplace_back(event);
}

void Patch::updatePatchRect()
{
	auto warpInv = warp_.inverse().matrix2x3();

	auto newCenterX = warpInv(0, 0) * (initPoint_.x) +
					  warpInv(0, 1) * (initPoint_.y) + warpInv(0, 2);
	auto newCenterY = warpInv(1, 0) * (initPoint_.x) +
					  warpInv(1, 1) * (initPoint_.y) + warpInv(1, 2);

	int extentX = (patch_.width - 1) / 2;
	int extentY = (patch_.height - 1) / 2;
	patch_ = cv::Rect2d((newCenterX - extentX), (newCenterY - extentY),
						2 * extentX + 1, 2 * extentY + 1);
}

void Patch::integrateEvents()
{
	if (events_.size() >= numOfEvents_)
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
		currentTimestamp_ = common::timestamp_t(static_cast<int32_t>(
			(events_.front().timestamp + events_.back().timestamp).count() *
			0.5));
	}
}

void Patch::warpImage(const cv::Mat& gradX, const cv::Mat& gradY)
{
	cv::Mat warpedGradX;
	cv::Mat warpedGradY;

	cv::Mat warpCv;
	cv::eigen2cv(warp_.matrix2x3(), warpCv);

	cv::warpAffine(gradX, warpedGradX, warpCv, {gradX.cols, gradX.rows},
				   cv::WARP_INVERSE_MAP);
	cv::warpAffine(gradY, warpedGradY, warpCv, {gradY.cols, gradY.rows},
				   cv::WARP_INVERSE_MAP);

	if (patch_.x < 0 || patch_.y < 0 || patch_.x + patch_.width >= gradX.cols ||
		patch_.y + patch_.height >= gradX.rows)
	{
		return;
	}

	predictedNabla_ = -warpedGradX(patch_) * std::cos(flowDir_) -
					  warpedGradY(patch_) * std::sin(flowDir_);
}

cv::Mat Patch::getNormalizedIntegratedNabla() const
{
	return integratedNabla_ / cv::norm(integratedNabla_);
}

void Patch::resetBatch()
{
	events_.clear();
}

Corner Patch::toCorner() const
{
	return cv::Point2d(patch_.x + (patch_.width - 1) / 2,
					   patch_.y + (patch_.height - 1) / 2);
}

bool Patch::isInPatch(const common::Point2i& point) const
{
	return patch_.contains(point);
}

common::Point2i Patch::patchToFrameCoords(
	const common::Point2i& pointInPatch) const
{
	return common::Point2i(pointInPatch.x + patch_.tl().x,
						   pointInPatch.y + patch_.tl().y);
}

common::Point2i Patch::frameToPatchCoords(
	const common::Point2i& pointInFrame) const
{
	return common::Point2i(pointInFrame.x - patch_.tl().x,
						   pointInFrame.y - patch_.tl().y);
}

common::EventSequence const& Patch::getEvents() const
{
	return events_;
}

void Patch::setWarp(const common::Pose2d& warp)
{
	warp_ = warp;
	init_ = true;
}

void Patch::setFlowDir(const double flowDir)
{
	flowDir_ = flowDir;
	init_ = true;
}

void Patch::setNumOfEvents(size_t numOfEvents)
{
	numOfEvents_ = std::max(minNumOfEvents_, numOfEvents);
	numOfEvents_ = std::min(numOfEvents_, maxNumOfEvents_);
}

void Patch::setIntegratedNabla(const cv::Mat& integratedNabla)
{
	integratedNabla_ = integratedNabla;
}

std::vector<common::Sample<common::Point2d>> const& Patch::getTrajectory() const
{
	return trajectory_;
}

void Patch::addTrajectoryPosition()
{
	trajectory_.push_back({toCorner(), currentTimestamp_});
}

void Patch::setCorner(const Corner& corner,
					  const common::timestamp_t& timestamp)
{
	patch_ = cv::Rect2d(corner.x - (patch_.width - 1) / 2,
						corner.y - (patch_.height - 1) / 2, patch_.width,
						patch_.height);
	initPoint_ = toCorner();
	init_ = false;
	currentTimestamp_ = timestamp;
	addTrajectoryPosition();
	resetBatch();
}

cv::Rect2d Patch::getInitPatch() const
{
	return cv::Rect2d(initPoint_.x - (patch_.width - 1) / 2,
					  initPoint_.y - (patch_.height - 1) / 2, patch_.width,
					  patch_.height);
}

}  // namespace tracker