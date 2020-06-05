#pragma once

#include <common/data_types.h>
#include "feature_tracker/patch.h"

namespace tracker
{
struct DetectorParams
{
	double qualityLevel = 0.01;
	double minDistance = 3;
	double associationDistance = 5;
	int32_t patchExtent = 17;
	int32_t blockSize = 3;
	cv::Size imageSize = {240, 180};
};

class FeatureDetector
{
   public:
	FeatureDetector(const DetectorParams& params);

	void extractPatches(const cv::Mat& image);

	Corners detectFeatures(const cv::Mat& image);

	void updatePatches(const common::EventSample& event);

	void associatePatches(Patches& newPatches);

	void setPatches(const Patches& patches) { patches_ = patches; }
	void setParams(const tracker::DetectorParams& params);
	void setTrackId(TrackId trackId) { nextTrackId_ = trackId; }

	Patches const& getPatches() const { return patches_; }
	Patches& getPatches() { return patches_; }
	Corners const& getFeatures() const { return corners_; }

   private:
	cv::Mat getLogImage(const cv::Mat& image);
	cv::Mat getGradients(const cv::Mat& image, bool xDir);
	void reset();

   private:
	DetectorParams params_;
	size_t maxCorners_;
	cv::Mat mask_;
	Patches patches_;
	Corners corners_;
	size_t nextTrackId_;
};
}  // namespace tracker