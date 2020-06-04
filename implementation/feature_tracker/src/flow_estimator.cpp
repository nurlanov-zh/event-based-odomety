#include "feature_tracker/flow_estimator.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace tracker
{
FlowEstimator::FlowEstimator(const FlowEstimatorParams& params)
	: imageCounter_(0), params_(params)
{
}

void FlowEstimator::addImage(const cv::Mat& image)
{
	if (imageCounter_ < 2) {
		imageCounter_++;
	}

	previousImage_ = std::move(currentImage_);
	currentImage_  = image;
}

bool FlowEstimator::getFlowPatches(Patches& patches)
{
	if (imageCounter_ != 2) {
		return false;
	}

    for (auto& patch : patches)
    {
        const auto& corner = patch.toCorner();

        const auto nextPoint = getFlow(corner);

        if (!nextPoint.has_value()) {
            continue;
        }

        // TODO check if patch is lost
        // TODO set initialized value

        // Since the velocity is normalized we need to store only angle
        const auto dirX = nextPoint.value().x - corner.x;
        const auto dirY = nextPoint.value().y - corner.y;
        const auto flowDir = std::atan2(dirY, dirX);

        patch.setFlowDir(flowDir);

        common::Pose2d warp;
        warp.translation() = Eigen::Vector2d(dirX, dirY);
        patch.setWarp(warp);
    }
    return true;
}

std::optional<cv::Point2f> FlowEstimator::getFlow(const cv::Point2f& curPoint)
{
    std::vector<cv::Point2f> curPoints = {curPoint};
    std::vector<cv::Point2f> nextPoints;
	std::vector<uint8_t> status;
	std::vector<float> error;

	// TODO what about using NN here to estimate flow
	cv::calcOpticalFlowPyrLK(previousImage_, currentImage_, curPoints,
							 nextPoints, status, error, params_.window,
							 params_.numLevels);

    // TODO check the error value

	if (status[0] == 0) {
        return {};
	}

	return std::make_optional(nextPoints[0]);
}
}  // ns tracker