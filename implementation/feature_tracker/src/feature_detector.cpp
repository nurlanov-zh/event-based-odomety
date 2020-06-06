#include "feature_tracker/feature_detector.h"

namespace tracker
{
FeatureDetector::FeatureDetector(const DetectorParams& params,
								 const cv::Size& imageSize)
	: params_(params)
{
	maxCorners_ =
		imageSize.width * imageSize.height /
		((2 * params_.patchExtent + 1) * (2 * params_.patchExtent + 1));

	mask_ = cv::Mat::zeros(imageSize, CV_8U);
	cv::rectangle(mask_,
				  {params_.patchExtent + 1, params_.patchExtent + 1,
				   imageSize.width - 2 * (params_.patchExtent + 1),
				   imageSize.height - 2 * (params_.patchExtent + 1)},
				  cvScalar(1), CV_FILLED);
}

void FeatureDetector::extractPatches(const cv::Mat& image)
{
	corners_ = detectFeatures(image);

	patches_.clear();
	patches_.reserve(corners_.size());
	for (const auto& corner : corners_)
	{
		patches_.emplace_back(Patch(corner, params_.patchExtent));
	}

	// TODO: build octree (e.g. with nanoflann) here and use it inside
	// updatePatches

	const auto& logImage = getLogImage(image);

	const auto gradX = getGradients(logImage, true);
	const auto gradY = getGradients(logImage, false);

	for (auto& patch : patches_)
	{
		patch.setGrad(gradX(patch.getPatch()), gradY(patch.getPatch()));
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
	}
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
	cv::Sobel(image, grad, CV_64F, xDir, !xDir, 3);
	return grad;
}

}  // namespace tracker