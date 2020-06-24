#pragma once

#include <common/data_types.h>
#include <feature_tracker/feature_detector.h>
#include <visual_odometry/visual_odometry.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>

namespace tools
{
struct EvaluatorParams
{
	cv::Size2i imageSize = {240, 180};
	std::string outputDir = "/tmp";
	common::CameraModelParams cameraModelParams = {};
	bool drawImages = false;
	bool experiment = true;
	// compensate whole image each second
	uint32_t compensationFrequencyTime = 1e6;
};

class Evaluator
{
   public:
	Evaluator(const EvaluatorParams& params);

	~Evaluator();

	void eventCallback(const common::EventSample& sample);

	void groundTruthCallback(const common::GroundTruthSample& sample);

	void imageCallback(const common::ImageSample& sample);

	void reset();

	void setTrackerParams(const tracker::DetectorParams& params);

	void saveTrajectory(const tracker::Patches& patches);

	void setGroundTruthSamples(const common::GroundTruth& groundTruthSamples);

	tracker::Patches const& getPatches() const;
	visual_odometry::MapLandmarks const& getMapLandmarks();
	std::list<visual_odometry::Keyframe> const& getActiveFrames() const;
	std::list<visual_odometry::Keyframe> const& getStoredFrames() const;

	cv::Mat const& getCompensatedEventImage();
	cv::Mat const& getIntegratedEventImage();

	void saveFinalCosts(
		const std::vector<tracker::OptimizerFinalLoss>& vectorFinalCosts);

   private:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	EvaluatorParams params_;

	std::unique_ptr<tracker::FeatureDetector> tracker_;
	std::unique_ptr<visual_odometry::VisualOdometryFrontEnd> visualOdometry_;

	tracker::Corners corners_;

	size_t imageNum_;
};
}  // namespace tools