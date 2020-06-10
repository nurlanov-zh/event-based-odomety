#pragma once

#include <common/data_types.h>
#include <feature_tracker/feature_detector.h>
#include <feature_tracker/flow_estimator.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>

namespace tools
{
struct EvaluatorParams
{
	cv::Size2i imageSize = {240, 180};
	std::string outputDir = "/tmp";
};

class Evaluator
{
   public:
	Evaluator(const EvaluatorParams& params);

	void eventCallback(const common::EventSample& sample);

	void groundTruthCallback(const common::GroundTruthSample& sample);

	void imageCallback(const common::ImageSample& sample);

	void reset();

	tracker::Patches const& getPatches() const;

	void setTrackerParams(const tracker::DetectorParams& params);

	void saveTrajectory(const tracker::Patches& patches);

   private:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	EvaluatorParams params_;

	std::unique_ptr<tracker::FeatureDetector> tracker_;
	std::unique_ptr<tracker::FlowEstimator> flowEstimator_;

	tracker::Corners corners_;

	size_t num_patches_;
};
}  // namespace tools