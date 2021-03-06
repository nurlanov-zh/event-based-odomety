#pragma once

#include "feature_tracker/patch.h"

#include <common/data_types.h>

#include <optional>

namespace tracker
{
struct FlowEstimatorParams
{
	int32_t numLevels = 3;
	cv::Size window = {21, 21};
	// seconds in double to microseconds
	double patchTimeWithoutUpdateScale = 1e6;
};

class FlowEstimator
{
   public:
	FlowEstimator(const FlowEstimatorParams& params);

	void addImage(const cv::Mat& image);

	bool getFlowPatches(Patches& patches);

   private:
	std::optional<cv::Point2f> getFlow(const cv::Point2f& curPoint);

   private:
   	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;
	
	size_t imageCounter_;
	cv::Mat previousImage_;
	cv::Mat currentImage_;

	FlowEstimatorParams params_;
};
}  // namespace tracker