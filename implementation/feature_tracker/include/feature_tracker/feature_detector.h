#pragma once

#include <common/data_types.h>
#include "feature_tracker/flow_estimator.h"
#include "feature_tracker/optimizer.h"
#include "feature_tracker/patch.h"

namespace tracker
{
struct DetectorParams
{
	double qualityLevel = 0.01;
	double minDistance = 3;
	double associationDistance = 10;
	int32_t patchExtent = 12;
	int32_t blockSize = 3;
	cv::Size imageSize = {240, 180};
	bool drawImages = false;
	OptimizerParams optimizerParams = {};
	int initNumEvents = 75;
};

class FeatureDetector
{
   public:
	FeatureDetector(const DetectorParams& params);

	void preExit();

	void newImage(const common::ImageSample& image);

	void extractPatches(const common::ImageSample& image);

	Corners detectFeatures(const cv::Mat& image);

	void updatePatches(const common::EventSample& event);

	void associatePatches(Patches& newPatches,
						  const common::timestamp_t& timestamp);

	void updateNumOfEvents(Patch& patch);

	void setPatches(const Patches& patches) { patches_ = patches; }
	void setParams(const tracker::DetectorParams& params);
	void setTrackId(TrackId trackId) { nextTrackId_ = trackId; }

	Patches const& getPatches() const { return patches_; }
	Patches& getPatches() { return patches_; }
	Corners const& getFeatures() const { return corners_; }
	Patches const& getArchivedPatches() const { return archivedPatches_; }

	std::vector<tracker::OptimizerFinalLoss> getOptimizedFinalCosts() const
	{
		return optimizer_->getFinalCosts();
	}

   private:
	cv::Mat getLogImage(const cv::Mat& image);
	cv::Mat getGradients(const cv::Mat& image, bool xDir);
	void init();
	void reset();

   private:
	cv::Mat gradX_;
	cv::Mat gradY_;

	std::unique_ptr<Optimizer> optimizer_;
	std::unique_ptr<tracker::FlowEstimator> flowEstimator_;

	DetectorParams params_;
	size_t maxCorners_;
	cv::Mat mask_;
	Patches patches_;
	Corners corners_;
	size_t nextTrackId_;
	Patches archivedPatches_;

	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;
};
}  // namespace tracker