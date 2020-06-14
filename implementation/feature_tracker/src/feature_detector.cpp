#include "feature_tracker/feature_detector.h"

#include <opencv2/core/eigen.hpp>

namespace tracker
{
FeatureDetector::FeatureDetector(const DetectorParams& params) : params_(params)
{
	init();
	reset();
}

void FeatureDetector::init()
{
	OptimizerParams params;
	params.drawCostMap = params_.drawImages;
	optimizer_.reset(new Optimizer(params, params_.imageSize));

	nextTrackId_ = 0;

	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");
}

void FeatureDetector::reset()
{
	maxCorners_ =
		params_.imageSize.width * params_.imageSize.height /
		((2 * params_.patchExtent + 1) * (2 * params_.patchExtent + 1));

	mask_ = cv::Mat::zeros(params_.imageSize, CV_8U);
	cv::rectangle(mask_,
				  {params_.patchExtent + 1, params_.patchExtent + 1,
				   params_.imageSize.width - 2 * (params_.patchExtent + 1),
				   params_.imageSize.height - 2 * (params_.patchExtent + 1)},
				  cvScalar(1), CV_FILLED);
}

void FeatureDetector::extractPatches(const common::ImageSample& image)
{

static int id = 0;
if (id < 2)
{
	corners_ = detectFeatures(image.value);
	id++;
	Patches newPatches;
	for (const auto& corner : corners_)
	{
		auto patch = Patch(corner, params_.patchExtent);
		patch.addTrajectoryPosition(corner, image.timestamp);
		newPatches.emplace_back(patch);
	}
	// patches_ = newPatches;

	const auto& logImage = getLogImage(image.value);

	gradX_ = getGradients(logImage, true);
	gradY_ = getGradients(logImage, false);
	optimizer_->setGrad(gradX_, gradY_);

	associatePatches(newPatches, image.timestamp);
	if (params_.drawImages)
	{
		for (auto& patch : patches_)
		{
			patch.warpImage(gradX_, gradY_);
		}
	}
}
}

Corners FeatureDetector::detectFeatures(const cv::Mat& image)
{
	Corners corners;

	cv::Mat imageGray = image;
	if (image.type() != CV_8U)
	{
		cv::cvtColor(image, imageGray, CV_RGB2GRAY);
	}

	cv::goodFeaturesToTrack(imageGray, corners, maxCorners_,
							params_.qualityLevel, params_.minDistance, mask_,
							params_.blockSize, true);

	return corners;
}

void FeatureDetector::updatePatches(const common::EventSample& event)
{
	for (auto& patch : patches_)
	{
		if (patch.isInPatch(event.value.point))
		{
			patch.addEvent(event);
		}

		if (patch.isReady() && patch.isInit() && !patch.isLost())
		{
			patch.integrateEvents();
			optimizer_->optimize(patch);
			updateNumOfEvents(patch);
			patch.resetBatch();
			patch.addTrajectoryPosition(patch.toCorner(), event.timestamp);
			if (params_.drawImages)
			{
				patch.warpImage(gradX_, gradY_);
			}
		}
	}
}

void FeatureDetector::associatePatches(Patches& newPatches,
									   const common::timestamp_t& timestamp)
{
	// greedy association...
	for (auto& patch : patches_)
	{
		const auto& corner = patch.toCorner();
		for (auto& newPatch : newPatches)
		{
			const auto& newCorner = newPatch.toCorner();
			if (newPatch.getTrackId() == -1 &&
				cv::norm(corner - newCorner) < params_.associationDistance)
			{
				// maybe update respective corner
				newPatch.setTrackId(patch.getTrackId());
				patch.setCorner(newPatch.toCorner());
				patch.addTrajectoryPosition(patch.toCorner(), timestamp);
				break;
			}
		}
	}

	for (auto& newPatch : newPatches)
	{
		if (newPatch.getTrackId() == -1)
		{
			newPatch.setTrackId(nextTrackId_);
			patches_.push_back(newPatch);
			nextTrackId_++;
		}
	}
}

void FeatureDetector::updateNumOfEvents(Patch& patch)
{
	const auto rect = patch.getPatch();
	if (rect.x < 0 || rect.y < 0 || rect.x + rect.width >= gradX_.cols ||
		rect.y + rect.height >= gradX_.rows)
	{
		return;
	}

	cv::Mat warpedGradX;
	cv::Mat warpedGradY;

	cv::Mat warpCv;
	cv::eigen2cv(patch.getWarp().matrix2x3(), warpCv);

	cv::warpAffine(gradX_, warpedGradX, warpCv, {gradX_.cols, gradX_.rows});
	cv::warpAffine(gradY_, warpedGradY, warpCv, {gradY_.cols, gradY_.rows});

	const auto gradX = warpedGradX(rect);
	const auto gradY = warpedGradY(rect);
	const auto flow = patch.getFlow();
	size_t sumPatch =
		cv::norm(gradX * std::cos(flow) + gradY * std::sin(flow), cv::NORM_L1);
	patch.setNumOfEvents(sumPatch);
	consoleLog_->debug(std::to_string(patch.getNumOfEvents()) +
					   " events are required for patch in track " +
					   std::to_string(patch.getTrackId()));
}

cv::Mat FeatureDetector::getLogImage(const cv::Mat& image)
{
	assert(image.type() == CV_8U);
	cv::Mat normalizedImage;
	cv::Mat logImage;

	image.convertTo(normalizedImage, CV_64F, 1.0 / 255.0);
	cv::log(normalizedImage + 10e-5, logImage);
	return logImage;
}

cv::Mat FeatureDetector::getGradients(const cv::Mat& image, bool xDir)
{
	assert(image.type() == CV_64F);

	cv::Mat grad;
	cv::Sobel(image / 8, grad, CV_64F, xDir, !xDir, 3);
	return grad;
}

void FeatureDetector::setParams(const tracker::DetectorParams& params)
{
	params_ = params;
	reset();
}

}  // namespace tracker