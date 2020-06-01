#pragma once

#include <common/data_types.h>
#include "feature_tracker/patch.h"

namespace tracker
{
struct DetectorParams
{
	double qualityLevel = 0.01;
	double minDistance  = 3;
	int32_t patchExtent = 17;
	int32_t blockSize   = 3;
};

class FeatureDetector
{
   public:
	FeatureDetector(const DetectorParams& params, const cv::Size& imageSize);

	void extractPatches(const cv::Mat& image);

	Corners detectFeatures(const cv::Mat& image);

	void updatePatches(const common::EventSample& event);

	void setPatches(const Patches& patches) { patches_ = patches; }

	Patches const& getPatches() const { return patches_; }
	Patches& getPatches() { return patches_; }

	Corners const& getFeatures() const { return corners_; }

   private:
	cv::Mat getLogImage(const cv::Mat& image);
	cv::Mat getGradients(const cv::Mat& image, bool xDir);

   private:
	DetectorParams params_;
	size_t maxCorners_;
	cv::Mat mask_;
	Patches patches_;
	Corners corners_;
};
}  // ns tracker