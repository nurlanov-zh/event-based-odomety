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
	common::CameraModelParams<double> cameraModelParams = {};
	bool drawImages = false;
	// compensate whole image each k microseconds
	uint32_t compensationFrequencyTime = 20000;
	uint32_t compensationFrequencyEvents = 15000;
	bool trackerExperiment = true;
	bool visOdometryExperiment = false;
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

	void saveFeaturesTrajectory(const tracker::Patches& patches);

	void savePoses(const std::list<visual_odometry::Keyframe>& keyframes);

	void saveGt(const std::vector<common::Pose3d>& gts);

	void saveMap(
		const std::vector<std::pair<tracker::TrackId, Eigen::Vector3d>>& map);

	void setGroundTruthSamples(const common::GroundTruth& groundTruthSamples);

	void setPatches(const tracker::Patches& patches);

	void setParams(const EvaluatorParams& params) { params_ = params; }

	tracker::Patches const& getPatches() const;
	visual_odometry::MapLandmarks const& getMapLandmarks();
	std::vector<std::pair<tracker::TrackId, Eigen::Vector3d>> const&
	getStoredMapLandmarks() const;
	std::map<size_t, visual_odometry::Keyframe> const& getActiveFrames() const;
	std::list<visual_odometry::Keyframe> const& getStoredFrames() const;
	std::vector<common::Pose3d> const& getGtPoses() const;

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

	std::unordered_map<size_t, tracker::Patches> keyframes_;

	tracker::Corners corners_;
	tracker::Patches patches_;

	size_t imageNum_;
};
}  // namespace tools