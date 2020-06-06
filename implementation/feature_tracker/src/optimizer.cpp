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

	gradGrid_.reset(new Grid(grad_.data(), 0, gradX.rows, 0, gradX.cols));
	gradInterpolator_.reset(new Interpolator(*gradGrid_));
}

void Optimizer::drawCostMap(const cv::Rect& rect,
							tracker::OptimizerCostFunctor* c)
{
	cv::Mat costImage = cv::Mat::zeros(rect.height, rect.width, CV_64F);
	for (int x = -(rect.width - 1) / 2; x <= (rect.width - 1) / 2; ++x)
	{
		for (int y = -(rect.width - 1) / 2; y <= (rect.width - 1) / 2; ++y)
		{
			common::Pose2d pose;
			pose.translation() = Eigen::Vector2d((float)x, (float)y);
			cv::Mat image = cv::Mat::zeros(rect.height, rect.width, CV_64F);
			(*c)(pose.data(), &flowDir_, (double*)image.data);
			double sum = 0;
			for (int xx = 0; xx < rect.width; ++xx)
			{
				for (int yy = 0; yy < rect.width; ++yy)
				{
					sum += std::pow(image.at<double>(yy, xx), 2);
				}
			}
			costImage.at<double>(y + (rect.width - 1) / 2,
								 x + (rect.width - 1) / 2) = sum;
		}
	}
	costImage_ = costImage;
}

void Optimizer::optimize(const Patch& patch)
{
	const auto rect = patch.getPatch();
	int size = rect.height * rect.width;

	const cv::Mat normalizedIntegratedNabla =
		patch.getNormalizedIntegratedNabla();

	warp_ = patch.getWarp();
	flowDir_ = patch.getFlow();

	ceres::Problem problem;

	problem.AddParameterBlock(warp_.data(), Sophus::SE2d::num_parameters,
							  new Sophus::test::LocalParameterizationSE2());
	problem.AddParameterBlock(&flowDir_, 1);

	auto* c = new tracker::OptimizerCostFunctor(normalizedIntegratedNabla,
												gradInterpolator_, rect);

	ceres::CostFunction* cost_function =
		new ceres::AutoDiffCostFunction<tracker::OptimizerCostFunctor,
										ceres::DYNAMIC,
										Sophus::SE2d::num_parameters, 1>(c,
																		 size);

	problem.AddResidualBlock(cost_function, NULL, warp_.data(), &flowDir_);

	if (params_.drawCostMap)
	{
		drawCostMap(rect, c);
	}
	// Set solver options (precision / method)
	ceres::Solver::Options options;

	options.minimizer_progress_to_stdout = false;
	options.num_threads = params_.numThreads;
	options.logging_type = ceres::SILENT;

	options.linear_solver_type = ceres::DENSE_QR;
	options.use_nonmonotonic_steps = true;
	options.max_num_consecutive_invalid_steps = 15;
	options.max_num_iterations = params_.maxNumIterations;
	options.function_tolerance = 1e-12;
	options.gradient_tolerance = 1e-12;

	// Solve
	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);

	consoleLog_->debug(summary.BriefReport());

	flowDir_ = fmod(flowDir_, 2 * M_PI);
}
}  // namespace tracker
