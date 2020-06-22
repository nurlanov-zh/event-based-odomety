#include "feature_tracker/optimizer.h"
#include "feature_tracker/local_parameterization_se2.hpp"

namespace tracker
{
Optimizer::Optimizer(const OptimizerParams& params, const cv::Size2i& imageSize)
	: params_(params), imageSize_(imageSize)
{
	grad_.resize(imageSize.height * imageSize.width * 2);
	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");
};

void Optimizer::setGrad(const cv::Mat& gradX, const cv::Mat& gradY)
{
	for (int row = 0; row < gradX.rows; row++)
	{
		for (int col = 0; col < gradX.cols; col++)
		{
			grad_[2 * row * imageSize_.width + 2 * col] =
				gradX.at<double>(row, col);
			grad_[2 * row * imageSize_.width + 2 * col + 1] =
				gradY.at<double>(row, col);
		}
	}
	gradGrid_.reset(
		new Grid(grad_.data(), 0, imageSize_.height, 0, imageSize_.width));
	gradInterpolator_.reset(new Interpolator(*(gradGrid_.get())));
}

void Optimizer::drawCostMap(Patch& patch, tracker::OptimizerCostFunctor* c)
{
	const auto rect = patch.getPatch();
	const common::Pose2d pose = patch.getWarp();
	double flowDir = patch.getFlow();

	int costMapWidth = params_.costMapWidth;
	int costMapHeight = params_.costMapHeight;

	cv::Mat costMap = cv::Mat::zeros(costMapHeight, costMapWidth, CV_64F);
	for (int x = -(costMapWidth - 1) / 2; x <= (costMapWidth - 1) / 2; ++x)
	{
		for (int y = -(costMapHeight - 1) / 2; y <= (costMapHeight - 1) / 2;
			 ++y)
		{
			const common::Pose2d poseNew(
				pose.log().z(),
				Eigen::Vector2d(
					static_cast<float>(x) + pose.matrix2x3()(0, 2),
					static_cast<float>(y) + pose.matrix2x3()(1, 2)));
			cv::Mat image = cv::Mat::zeros(rect.height, rect.width, CV_64F);
			(*c)(poseNew.data(), &flowDir, (double*)image.data);
			double sum = cv::norm(image, cv::NORM_L2);
			costMap.at<double>(y + (costMapHeight - 1) / 2,
							   x + (costMapWidth - 1) / 2) = sum;
		}
	}
	patch.setCostMap(costMap);
}

void Optimizer::optimize(Patch& patch)
{
	// update currentTimestamp and timeLastUpdate inside
	patch.integrateEvents();

	consoleLog_->debug("Optimizing... patch number " +
					   std::to_string(patch.getTrackId()));
	std::chrono::steady_clock::time_point begin =
		std::chrono::steady_clock::now();

	const cv::Rect2d currentRect = patch.getPatch();
	int size = currentRect.height * currentRect.width;

	const cv::Mat normalizedIntegratedNabla =
		patch.getNormalizedIntegratedNabla();

	auto warp = patch.getWarp();
	double flowDir = patch.getFlow();

	ceres::Problem problem;

	problem.AddParameterBlock(warp.data(), Sophus::SE2d::num_parameters,
							  new Sophus::test::LocalParameterizationSE2());
	problem.AddParameterBlock(&flowDir, 1);

	auto* c = new tracker::OptimizerCostFunctor(normalizedIntegratedNabla,
												gradInterpolator_.get(),
												currentRect, imageSize_);

	ceres::CostFunction* cost_function =
		new ceres::AutoDiffCostFunction<tracker::OptimizerCostFunctor,
										ceres::DYNAMIC,
										Sophus::SE2d::num_parameters, 1>(c,
																		 size);

	problem.AddResidualBlock(cost_function,
							 new ceres::HuberLoss(params_.huberLoss),
							 warp.data(), &flowDir);

	// Set solver options (precision / method)
	ceres::Solver::Options options;

	options.minimizer_progress_to_stdout = false;
	options.num_threads = params_.numThreads;
	options.logging_type = ceres::SILENT;

	options.linear_solver_type = ceres::DENSE_QR;
	options.use_nonmonotonic_steps = true;
	options.max_num_iterations = params_.maxNumIterations;

	// Solve
	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);

	consoleLog_->debug(summary.BriefReport());

	std::chrono::steady_clock::time_point end =
		std::chrono::steady_clock::now();
	consoleLog_->debug(
		"Optimiztion elapsed TIME: " +
		std::to_string(
			std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
				.count()) +
		" [ms]\n");

	patch.addFinalCost(summary.final_cost);

	tracker::OptimizerFinalLoss finalLoss = {
		patch.getTrackId(),
		summary.final_cost,
		patch.getCurrentTimestamp().count(),
	};
	vectorFinalCost_.emplace_back(finalLoss);

	// filtering by costs
	const auto patchFinalCosts = patch.getFinalCosts();
	if (patchFinalCosts.size() >= 5)
	{
		std::vector<double> lastFiveCosts;
		for (int i = 0; i < 5; i++)
		{
			lastFiveCosts.push_back(
				patchFinalCosts[patchFinalCosts.size() - 6 + i]);
		}
		std::sort(lastFiveCosts.begin(), lastFiveCosts.end());
		double median = lastFiveCosts[2];

		if (median > params_.optimizerThreshold)
		{
			consoleLog_->info("Patch: " + std::to_string(patch.getTrackId()) +
							  " lost with median last 5 final cost: " +
							  std::to_string(median));
			patch.setLost();
			return;
		}
	}

	// !!! Update patch params !!!
	consoleLog_->debug("Updating patch: " + std::to_string(patch.getTrackId()));

	consoleLog_->debug("Old center: (" +
					   std::to_string(int(patch.toCorner().x)) + ", " +
					   std::to_string(int(patch.toCorner().y)) + ")");

	flowDir = fmod(flowDir, 2 * M_PI);
	patch.setFlowDir(flowDir);
	patch.setWarp(warp);

	auto oldCenter = patch.toCorner();
	patch.updatePatchRect();
	auto newCenter = patch.toCorner();
	auto newTimeWithoutUpdate = common::timestamp_t(
		static_cast<int64_t>(params_.patchTimeWithoutUpdateScale /
							 fmax(1e-1, cv::norm(newCenter - oldCenter))));

	patch.setTimeWithoutUpdate(newTimeWithoutUpdate);

	patch.addTrajectoryPosition();

	// Need events here!
	patch.integrateMotionCompensatedEvents();

	// Now can erase events.
	patch.resetBatch();

	consoleLog_->debug("New center: (" +
					   std::to_string(int(patch.toCorner().x)) + ", " +
					   std::to_string(int(patch.toCorner().y)) + ")");

	if (params_.drawCostMap)
	{
		consoleLog_->debug("Drawing costmap...");
		begin = std::chrono::steady_clock::now();
		// it is too slow
		drawCostMap(patch, c);
		end = std::chrono::steady_clock::now();
		consoleLog_->debug(
			"Costmap drawing finished in " +
			std::to_string(
				std::chrono::duration_cast<std::chrono::milliseconds>(end -
																	  begin)
					.count()) +
			"[ms]");
	}
}
}  // namespace tracker
