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
	double minDistance = 10;
	double associationDistance = 5;
	int32_t patchExtent = 12;
	int32_t blockSize = 3;
	cv::Size imageSize = {240, 180};
	bool drawImages = false;
	OptimizerParams optimizerParams = {};
	int initNumEvents = 75;
	u_long maxNumEventsToStore = 15000;
	bool useAverageFlow = true;
	bool optimizeFlowTV = true;
	bool useL1 = false;
	cv::Size patchCompensateSize = {20, 20};
	double compensateTVweight = 1e3;
	double compensateTVHuberLoss = 10;
	double compensateScale = 1e-3;
	uint compensateMinNumEvents = 100;
	size_t maxPatches = 100;
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

	void addEvent(const common::EventSample& event);

	void initMotionField(const common::timestamp_t timestamp);

	void interpolateMotionField(const common::timestamp_t timestamp);

	void compensateEvents(const std::list<common::EventSample>& events);

	void compensateEventsContrast(const std::list<common::EventSample>& events);

	void clearEvents() { lastEvents_.clear(); }

	void integrateEvents(const std::list<common::EventSample>& events);

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
	std::list<common::EventSample> const& getEvents() { return lastEvents_; }
	cv::Mat const& getCompensatedEventImage();
	cv::Mat const& getIntegratedEventImage();
	common::timestamp_t const& getLastCompensation();

	std::vector<tracker::OptimizerFinalLoss> getOptimizedFinalCosts() const
	{
		return optimizers_.begin()->second->getFinalCosts();
	}

   private:
	cv::Mat getLogImage(const cv::Mat& image);
	cv::Mat getGradients(const cv::Mat& image, bool xDir);
	void init();
	void reset();

   private:
	cv::Mat gradX_;
	cv::Mat gradY_;

	cv::Mat compensatedEventImage_;

	cv::Mat integratedEventImage_;

	cv::Mat motionField_;
	std::vector<common::Point2i> fixedPoints_;

	std::unique_ptr<Optimizer> optimizer_;
	std::unique_ptr<tracker::FlowEstimator> flowEstimator_;

	std::unordered_map<size_t, std::unique_ptr<Optimizer>> optimizers_;

	DetectorParams params_;
	size_t maxCorners_;
	cv::Mat mask_;
	Patches patches_;
	Corners corners_;
	size_t nextTrackId_;
	Patches archivedPatches_;

	std::list<common::EventSample> lastEvents_;
	common::timestamp_t lastCompensation;

	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;
};
}  // namespace tracker