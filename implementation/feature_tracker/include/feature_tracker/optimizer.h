#pragma once

#include "feature_tracker/optimizer_cost.h"
#include "feature_tracker/patch.h"

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

namespace tracker
{
struct OptimizerParams
{
	bool drawCostMap = false;
	int maxNumIterations = 30;
	int numThreads = 1;
};

class Optimizer
{
   public:
	Optimizer(const OptimizerParams& params, const cv::Size2i& imageSize);

	void optimize(const Patch& patch);

	common::Pose2d getWarp() const { return warp_; }

	double getFlowDir() const { return flowDir_; }

	const cv::Mat& getCostImage() const { return costImage_; }

	void setGrad(const cv::Mat& gradX, const cv::Mat& gradY);

	void setWarp(const common::Pose2d& warp) { warp_ = warp; }

	void setFlowDir(const double& flowDir) { flowDir_ = flowDir; }

   private:
	void drawCostMap(const cv::Rect& rect, tracker::OptimizerCostFunctor* c);

   private:
	OptimizerParams params_;
	common::Pose2d warp_;
	double flowDir_;
	cv::Size2i imageSize_;
	cv::Mat costImage_;

	std::vector<double> grad_;
	GridPtr gradGrid_;
	InterpolatorPtr gradInterpolator_;

	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;
};

}  // namespace tracker