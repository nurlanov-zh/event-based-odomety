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
	//	int numThreads = tbb::task_scheduler_init::default_num_threads();
	int numThreads = 1;
	double optimizerThreshold = 0.25;
	double huberLoss = 0.3;
	int costMapWidth = 11;
	int costMapHeight = 11;
	// seconds in double to microseconds
	double patchTimeWithoutUpdateScale = 1e6;
};

struct OptimizerFinalLoss
{
	tracker::TrackId trackId;
	double lossValue;
	int64_t timeStampMicrosecond;
};

class Optimizer
{
   public:
	Optimizer(const OptimizerParams& params, const cv::Size2i& imageSize);

	void optimize(Patch& patch);

	void setGrad(const cv::Mat& gradX, const cv::Mat& gradY);

	OptimizerParams* getParams() { return &params_; }

	std::vector<OptimizerFinalLoss> getFinalCosts() { return vectorFinalCost_; }

	void setParams(const OptimizerParams& params) { params_ = params; }

   private:
	void drawCostMap(Patch& patch, tracker::OptimizerCostFunctor* c);

   private:
	OptimizerParams params_;
	cv::Size2i imageSize_;

	std::vector<double> grad_;
	GridPtr gradGrid_;
	InterpolatorPtr gradInterpolator_;

	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	std::vector<OptimizerFinalLoss> vectorFinalCost_;
};

}  // namespace tracker