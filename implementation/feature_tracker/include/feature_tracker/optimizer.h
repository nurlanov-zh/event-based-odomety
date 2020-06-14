#pragma once

#include "feature_tracker/optimizer_cost.h"
#include "feature_tracker/patch.h"

#include <common/data_types.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <tbb/task_scheduler_init.h>

namespace tracker
{
struct OptimizerParams
{
	bool drawCostMap = false;
	int maxNumIterations = 50;
	int numThreads = tbb::task_scheduler_init::default_num_threads();
	;
};

class Optimizer
{
   public:
	Optimizer(const OptimizerParams& params, const cv::Size2i& imageSize);

	void optimize(Patch& patch);

	void setGrad(const cv::Mat& gradX, const cv::Mat& gradY);

   private:
	void drawCostMap(Patch& patch, tracker::OptimizerCostFunctor* c,
					 bool first);

   private:
	OptimizerParams params_;
	cv::Size2i imageSize_;

	std::vector<double> grad_;
	GridPtr gradGrid_;
	InterpolatorPtr gradInterpolator_;

	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;
};

}  // namespace tracker