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

void Optimizer::drawCostMap(Patch& patch, tracker::OptimizerCostFunctor* c,
							bool first)
{
	const auto rect = patch.getPatch();
	const common::Pose2d pose = patch.getWarp();
	double flowDir = patch.getFlow();

	cv::Mat costMap = cv::Mat::zeros(rect.height, rect.width, CV_64F);
	for (int x = -(rect.width - 1) / 2; x <= (rect.width - 1) / 2; ++x)
	{
		for (int y = -(rect.width - 1) / 2; y <= (rect.width - 1) / 2; ++y)
		{
			const common::Pose2d poseNew(
				pose.log().z(),
				Eigen::Vector2d(
					static_cast<float>(x) + pose.matrix2x3()(0, 2),
					static_cast<float>(y) + pose.matrix2x3()(1, 2)));
			cv::Mat image = cv::Mat::zeros(rect.height, rect.width, CV_64F);
			(*c)(poseNew.data(), &flowDir, (double*)image.data);
			double sum = 0;
			for (int xx = 0; xx < rect.width; ++xx)
			{
				for (int yy = 0; yy < rect.width; ++yy)
				{
					sum += std::pow(image.at<double>(yy, xx), 2);
				}
			}
			costMap.at<double>(y + (rect.width - 1) / 2,
							   x + (rect.width - 1) / 2) = sum;
		}
	}
	if (first)
	{
		patch.setCostMap(costMap);
	}
	else
	{
		patch.setCostMap2(costMap);
	}
}

void Optimizer::optimize(Patch& patch)
{
	const cv::Rect2i initRect = patch.getInitPatch();
	int size = initRect.height * initRect.width;

	const cv::Mat normalizedIntegratedNabla =
		patch.getNormalizedIntegratedNabla();

	auto warp = patch.getWarp();
	double flowDir = patch.getFlow();

	ceres::Problem problem;

	problem.AddParameterBlock(warp.data(), Sophus::SE2d::num_parameters,
							  new Sophus::test::LocalParameterizationSE2());
	problem.AddParameterBlock(&flowDir, 1);

	auto* c = new tracker::OptimizerCostFunctor(
		normalizedIntegratedNabla, gradInterpolator_.get(), initRect);

	ceres::CostFunction* cost_function =
		new ceres::AutoDiffCostFunction<tracker::OptimizerCostFunctor,
										ceres::DYNAMIC,
										Sophus::SE2d::num_parameters, 1>(c,
																		 size);

	problem.AddResidualBlock(cost_function, NULL, warp.data(), &flowDir);

//	if (params_.drawCostMap)
//	{
//		// it is too slow
//		drawCostMap(patch, c, true);
//	}

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

	if (summary.final_cost > 0.9)
	{
		patch.setLost();
		return;
	}

	// !!! Update patch params !!!
	consoleLog_->debug(
		"New warp is (" + std::to_string(warp.translation().x()) + ", " +
		std::to_string(warp.translation().y()) + ", " +
		std::to_string(warp.log().z()) + ") vs (" +
		std::to_string(patch.getWarp().translation().x()) + ", " +
		std::to_string(patch.getWarp().translation().y()) + ", " +
		std::to_string(patch.getWarp().log().z()) + ")");

	consoleLog_->debug("New flow is " + std::to_string(flowDir) + " vs " +
					   std::to_string(patch.getFlow()));

	flowDir = fmod(flowDir, 2 * M_PI);
	patch.setFlowDir(flowDir);
	patch.setWarp(warp);
	patch.updatePatchRect(warp);

//	if (params_.drawCostMap)
//	{
//		// it is too slow
//		drawCostMap(patch, c, false);
//	}
}
}  // namespace tracker
