#include "feature_tracker/feature_detector.h"
#include "feature_tracker/contrast_functor.h"
#include "feature_tracker/total_variance.h"

#include <opencv2/core/eigen.hpp>

namespace tracker
{
FeatureDetector::FeatureDetector(const DetectorParams& params) : params_(params)
{
	init();
	reset();
}

void FeatureDetector::init()
{
	OptimizerParams params;
	params = params_.optimizerParams;
	optimizer_.reset(new Optimizer(params, params_.imageSize));

	flowEstimator_.reset(
		new tracker::FlowEstimator(tracker::FlowEstimatorParams()));

	nextTrackId_ = 0;

	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");
	initMotionField(common::timestamp_t(0));
	compensatedEventImage_ = cv::Mat::zeros(params_.imageSize, CV_64F);
	integratedEventImage_ = cv::Mat::zeros(params_.imageSize, CV_64F);
	lastCompensation = common::timestamp_t(0);
	lastEvents_.clear();
	fixedPoints_.clear();
}

void FeatureDetector::reset()
{
	maxCorners_ =
		params_.imageSize.width * params_.imageSize.height /
		((2 * params_.patchExtent + 1) * (2 * params_.patchExtent + 1));

	mask_ = cv::Mat::zeros(params_.imageSize, CV_8U);
	cv::rectangle(mask_,
				  {params_.patchExtent + 1, params_.patchExtent + 1,
				   params_.imageSize.width - 2 * (params_.patchExtent + 1),
				   params_.imageSize.height - 2 * (params_.patchExtent + 1)},
				  cvScalar(1), CV_FILLED);

	initMotionField(common::timestamp_t(0));
	compensatedEventImage_ = cv::Mat::zeros(params_.imageSize, CV_64F);
	integratedEventImage_ = cv::Mat::zeros(params_.imageSize, CV_64F);
	lastCompensation = common::timestamp_t(0);
	lastEvents_.clear();
	fixedPoints_.clear();
}

void FeatureDetector::initMotionField(const common::timestamp_t timestamp)
{
	motionField_ = cv::Mat::zeros(params_.imageSize, CV_64FC2);
	fixedPoints_.clear();
	double avgX = 0.0;
	double avgY = 0.0;
	double count = 0.0;
	if (timestamp.count() > 0 && !patches_.empty())
	{
		for (const auto& patch : patches_)
		{
			// Find left border in patch trajectory according to timestamp
			// using Binary Search assuming that trajectory is sorted.
			auto low =
				std::lower_bound(patch.getTrajectory().begin(),
								 patch.getTrajectory().end(), timestamp,
								 [](common::Sample<common::Point2d> const& lhs,
									common::timestamp_t const& target) {
									 return lhs.timestamp < target;
								 });
			if (low != patch.getTrajectory().end() and
				(low + 1) != patch.getTrajectory().end())
			{
				const auto oldP =
					common::Point2i(round(low->value.x), round(low->value.y));

				motionField_.at<cv::Vec2f>(oldP.y, oldP.x)[0] =
					1e3 * ((low + 1)->value.x - low->value.x) /
					static_cast<double>(
						((low + 1)->timestamp - low->timestamp).count());

				motionField_.at<cv::Vec2f>(oldP.y, oldP.x)[1] =
					1e3 * ((low + 1)->value.y - low->value.y) /
					static_cast<double>(
						((low + 1)->timestamp - low->timestamp).count());

				fixedPoints_.emplace_back(oldP);

				avgX += motionField_.at<cv::Vec2f>(oldP.y, oldP.x)[0];
				avgY += motionField_.at<cv::Vec2f>(oldP.y, oldP.x)[1];
				count += 1.0;
			}
		}

		// Init the rest with an average flow or as the flow of neighbour
		if (!fixedPoints_.empty())
		{
			for (int y = 0; y < motionField_.rows; y++)
			{
				for (int x = 0; x < motionField_.cols; x++)
				{
					if (motionField_.at<cv::Vec2f>(y, x)[0] == 0 and
						motionField_.at<cv::Vec2f>(y, x)[1] == 0)
					{
						if (params_.useAverageFlow)
						{
							motionField_.at<cv::Vec2f>(y, x)[0] = avgX / count;
							motionField_.at<cv::Vec2f>(y, x)[1] = avgY / count;
						}
						else
						{
							int bestId = 0;
							double bestDist = 1e16;
							for (int id = 0;
								 id < static_cast<int>(fixedPoints_.size());
								 id++)
							{
								const auto& point = fixedPoints_[id];
								double dist = (x - point.x) * (x - point.x) +
											  (y - point.y) * (y - point.y);
								if (dist < bestDist)
								{
									bestDist = dist;
									bestId = id;
								}
							}
							const auto& bestP = fixedPoints_[bestId];
							motionField_.at<cv::Vec2f>(y, x)[0] =
								motionField_.at<cv::Vec2f>(bestP.y, bestP.x)[0];
							motionField_.at<cv::Vec2f>(y, x)[1] =
								motionField_.at<cv::Vec2f>(bestP.y, bestP.x)[1];
						}
					}
				}
			}
		}
	}
}

void FeatureDetector::interpolateMotionField(
	const common::timestamp_t timestamp)
{
	// init motion field according to patches at timestamp, fill fixedPoints_
	initMotionField(timestamp);
	cv::Mat oldMotionField = motionField_.clone();

	// if initialized
	if (cv::norm(motionField_) > 0)
	{
		ceres::Problem problem;

		// init and fill data into array
		auto* mf =
			new double[params_.imageSize.width * params_.imageSize.height * 2];
		for (int y = 0; y < params_.imageSize.height; y++)
		{
			for (int x = 0; x < params_.imageSize.width; x++)
			{
				mf[2 * (y * params_.imageSize.width + x)] =
					motionField_.at<cv::Vec2f>(y, x)[0];
				mf[1 + 2 * (y * params_.imageSize.width + x)] =
					motionField_.at<cv::Vec2f>(y, x)[1];
			}
		}
		// Create optimization problem
		for (int y = 0; y < params_.imageSize.height - 1; y++)
		{
			for (int x = 0; x < params_.imageSize.width - 1; x++)
			{
				ceres::CostFunction* cost_function =
					new ceres::AutoDiffCostFunction<
						tracker::totalVarianceFunctor, 2, 2, 2>(
						new tracker::totalVarianceFunctor());

				if (params_.useL1)
				{
					problem.AddResidualBlock(
						cost_function, new ceres::HuberLoss(1e-5),
						&mf[2 * (y * params_.imageSize.width + x)],
						&mf[2 * (y * params_.imageSize.width + x + 1)]);

					problem.AddResidualBlock(
						cost_function, new ceres::HuberLoss(1e-5),
						&mf[2 * (y * params_.imageSize.width + x)],
						&mf[2 * ((y + 1) * params_.imageSize.width + x)]);
				}
				else
				{
					problem.AddResidualBlock(
						cost_function, nullptr,
						&mf[2 * (y * params_.imageSize.width + x)],
						&mf[2 * (y * params_.imageSize.width + x + 1)]);

					problem.AddResidualBlock(
						cost_function, nullptr,
						&mf[2 * (y * params_.imageSize.width + x)],
						&mf[2 * ((y + 1) * params_.imageSize.width + x)]);
				}
			}
		}

		for (const auto& point : fixedPoints_)
		{
			if (!problem.IsParameterBlockConstant(
					&mf[2 * (point.y * params_.imageSize.width + point.x)]))
			{
				problem.SetParameterBlockConstant(
					&mf[2 * (point.y * params_.imageSize.width + point.x)]);
			}
		}

		// Set solver options (precision / method)
		ceres::Solver::Options options;
		options.minimizer_progress_to_stdout = false;
		options.num_threads = 1;
		options.logging_type = ceres::SILENT;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.use_nonmonotonic_steps = false;

		// Solve
		ceres::Solver::Summary summary;
		Solve(options, &problem, &summary);

		consoleLog_->debug(summary.BriefReport());

		for (int y = 0; y < params_.imageSize.height; y++)
		{
			for (int x = 0; x < params_.imageSize.width; x++)
			{
				motionField_.at<cv::Vec2f>(y, x)[0] =
					mf[2 * (y * params_.imageSize.width + x)];
				motionField_.at<cv::Vec2f>(y, x)[1] =
					mf[1 + 2 * (y * params_.imageSize.width + x)];
			}
		}
	}
}

void FeatureDetector::compensateEvents(
	const std::list<common::EventSample>& events)
{
	// take mid timestamp
	const auto timestamp = common::timestamp_t(static_cast<int32_t>(
		(events.front().timestamp + events.back().timestamp).count() * 0.5));

	lastCompensation = events.back().timestamp;

	// Optimize Motion Field
	if (params_.optimizeFlowTV)
	{
		interpolateMotionField(timestamp);
	}
	else
	{
		initMotionField(timestamp);
	}

	consoleLog_->info(
		"Compensating {} events, to time_ref {}, with start "
		"at {}, end at {} microseconds, using {} fixed patch centers",
		events.size(), timestamp.count(), events.front().timestamp.count(),
		events.back().timestamp.count(), fixedPoints_.size());

	// Compensate events
	compensatedEventImage_ = cv::Mat::zeros(params_.imageSize, CV_64F);
	for (const auto& event : events)
	{
		// Simple Motion Compensation formula:
		// event_t = event_0 + (t - t_event) / (t_final - t_init) * dir
		common::Point2i newP;

		newP.x = round(event.value.point.x +
					   (timestamp - event.timestamp).count() * 1e-3 *
						   motionField_.at<cv::Vec2f>(event.value.point.y,
													  event.value.point.x)[0]);
		newP.y = round(event.value.point.y +
					   (timestamp - event.timestamp).count() * 1e-3 *
						   motionField_.at<cv::Vec2f>(event.value.point.y,
													  event.value.point.x)[1]);

		if (newP.x >= 0 and newP.x < compensatedEventImage_.cols and
			newP.y >= 0 and newP.y < compensatedEventImage_.rows)
		{
			//			compensatedEventImage_.at<double>(newP.y, newP.x) +=
			//				static_cast<double>(event.value.sign);
			compensatedEventImage_.at<double>(newP.y, newP.x) +=
				static_cast<double>(1);
		}
	}
}

void FeatureDetector::compensateEventsContrast(
	const std::list<common::EventSample>& events)
{
	int numPatchesX =
		params_.imageSize.width / params_.patchCompensateSize.width;
	int numPatchesY =
		params_.imageSize.height / params_.patchCompensateSize.height;
	const auto timestamp = common::timestamp_t(static_cast<int32_t>(
		(events.front().timestamp + events.back().timestamp).count() * 0.5));
	lastCompensation = events.back().timestamp;

	consoleLog_->info(
		"Compensating {} events, to time_ref {}, with start "
		"at {}, end at {} microseconds, using {}x{}-sized {}x{} patches",
		events.size(), timestamp.count(), events.front().timestamp.count(),
		events.back().timestamp.count(), params_.patchCompensateSize.width,
		params_.patchCompensateSize.height, numPatchesX, numPatchesY);

	ceres::Problem problem;
	// init and fill data into array
	auto* mf = new double[numPatchesX * numPatchesY * 2];
	for (int y = 0; y < numPatchesY; y++)
	{
		for (int x = 0; x < numPatchesX; x++)
		{
			mf[2 * (y * numPatchesX + x)] = 0;
			mf[2 * (y * numPatchesX + x) + 1] = 0;
		}
	}

	for (int y = 0; y < numPatchesY; y++)
	{
		for (int x = 0; x < numPatchesX; x++)
		{
			cv::Rect2i patchRect;
			patchRect.x = x * params_.patchCompensateSize.width;
			patchRect.y = y * params_.patchCompensateSize.height;
			patchRect.width = params_.patchCompensateSize.width;
			patchRect.height = params_.patchCompensateSize.height;
			if (x == numPatchesX - 1)
			{
				patchRect.width = params_.imageSize.width -
								  x * params_.patchCompensateSize.width;
			}
			if (y == numPatchesY - 1)
			{
				patchRect.height = params_.imageSize.height -
								   y * params_.patchCompensateSize.height;
			}

			std::list<common::EventSample> patchEvents;
			for (const auto& event : events)
			{
				if (patchRect.contains(event.value.point))
				{
					patchEvents.push_back(event);
				}
			}

			if (!patchEvents.empty())
			{
				ceres::CostFunction* cost_function =
					new ceres::AutoDiffCostFunction<tracker::contrastFunctor, 1,
													2>(
						new tracker::contrastFunctor(patchEvents, patchRect));

				problem.AddResidualBlock(cost_function, nullptr,
										 &mf[2 * (y * numPatchesX + x)]);
			}

			if (x < numPatchesX - 1)
			{
				ceres::CostFunction* cost_function =
					new ceres::AutoDiffCostFunction<
						tracker::totalVarianceFunctor, 2, 2, 2>(
						new tracker::totalVarianceFunctor(5e2));

				problem.AddResidualBlock(cost_function,
										 new ceres::HuberLoss(1e2),
										 &mf[2 * (y * numPatchesX + x)],
										 &mf[2 * (y * numPatchesX + x + 1)]);
			}
			if (y < numPatchesY - 1)
			{
				ceres::CostFunction* cost_function =
					new ceres::AutoDiffCostFunction<
						tracker::totalVarianceFunctor, 2, 2, 2>(
						new tracker::totalVarianceFunctor(5e2));

				problem.AddResidualBlock(cost_function,
										 new ceres::HuberLoss(1e2),
										 &mf[2 * (y * numPatchesX + x)],
										 &mf[2 * ((y + 1) * numPatchesX + x)]);
			}
		}
	}

	// Set solver options (precision / method)
	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = false;
	options.num_threads = 1;
	options.logging_type = ceres::SILENT;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.use_nonmonotonic_steps = true;
	options.max_num_iterations = 50;
	options.function_tolerance = 1e-12;
	options.gradient_tolerance = 1e-12;
	options.parameter_tolerance = 1e-12;

	// Solve
	ceres::Solver::Summary summary;
	Solve(options, &problem, &summary);

	consoleLog_->info(summary.BriefReport());

	for (int y = 0; y < numPatchesY; y++)
	{
		for (int x = 0; x < numPatchesX; x++)
		{
			motionField_.at<cv::Vec2f>(
				y * params_.patchCompensateSize.height,
				x * params_.patchCompensateSize.width)[0] =
				mf[2 * (y * numPatchesX + x)];
			motionField_.at<cv::Vec2f>(
				y * params_.patchCompensateSize.height,
				x * params_.patchCompensateSize.width)[1] =
				mf[2 * (y * numPatchesX + x) + 1];
		}
	}

	compensatedEventImage_ = cv::Mat::zeros(params_.imageSize, CV_64F);
	for (const auto& event : events)
	{
		int x = std::min(
			int(event.value.point.x / params_.patchCompensateSize.width),
			numPatchesX - 1);
		int y = std::min(
			int(event.value.point.y / params_.patchCompensateSize.height),
			numPatchesY - 1);
		// Simple Motion Compensation formula:
		// event_t = event_0 + (t - t_event) / (t_final - t_init) * dir
		common::Point2i newP;

		newP.x = round(event.value.point.x +
					   (timestamp - event.timestamp).count() * 1e-3 *
						   mf[2 * (y * numPatchesX + x)]);
		newP.y = round(event.value.point.y +
					   (timestamp - event.timestamp).count() * 1e-3 *
						   mf[2 * (y * numPatchesX + x) + 1]);

		if (newP.x >= 0 and newP.x < compensatedEventImage_.cols and
			newP.y >= 0 and newP.y < compensatedEventImage_.rows)
		{
			//			compensatedEventImage_.at<double>(newP.y, newP.x) +=
			//				static_cast<double>(event.value.sign);
			compensatedEventImage_.at<double>(newP.y, newP.x) +=
				static_cast<double>(1);
		}
	}
}

void FeatureDetector::integrateEvents(
	const std::list<common::EventSample>& events)
{
	integratedEventImage_ = cv::Mat::zeros(params_.imageSize, CV_64F);
	for (const auto& event : events)
	{
		const common::Point2i newP = event.value.point;
		if (newP.x >= 0 and newP.x < integratedEventImage_.cols and
			newP.y >= 0 and newP.y < integratedEventImage_.rows)
		{
			//			integratedEventImage_.at<double>(newP.y, newP.x) +=
			//				static_cast<double>(event.value.sign);
			integratedEventImage_.at<double>(newP.y, newP.x) +=
				static_cast<double>(1);
		}
	}
}

void FeatureDetector::preExit()
{
	for (auto patchIt = patches_.begin(); patchIt != patches_.end();)
	{
		archivedPatches_.push_back(*patchIt);
		++patchIt;
	}
}

void FeatureDetector::newImage(const common::ImageSample& image)
{
	extractPatches(image);

	flowEstimator_->addImage(image.value);
	flowEstimator_->getFlowPatches(patches_);
	for (auto& patch : patches_)
	{
		updateNumOfEvents(patch);
	}

	if (params_.drawImages)
	{
		for (auto& patch : patches_)
		{
			patch.warpImage(gradX_, gradY_);
		}
	}

	for (auto patchIt = patches_.begin(); patchIt != patches_.end();)
	{
		if (patchIt->isLost())
		{
			archivedPatches_.push_back(*patchIt);
			patchIt = patches_.erase(patchIt);
			continue;
		}
		++patchIt;
	}
	consoleLog_->info("Extracted " + std::to_string(patches_.size()) +
					  " patches.");
}

void FeatureDetector::extractPatches(const common::ImageSample& image)
{
	corners_ = detectFeatures(image.value);
	Patches newPatches;
	for (const auto& corner : corners_)
	{
		auto patch = Patch(corner, params_.patchExtent, image.timestamp);
		newPatches.emplace_back(patch);
	}

	const auto& logImage = getLogImage(image.value);

	gradX_ = getGradients(logImage, true);
	gradY_ = getGradients(logImage, false);
	optimizer_->setGrad(gradX_, gradY_);

	associatePatches(newPatches, image.timestamp);
}

Corners FeatureDetector::detectFeatures(const cv::Mat& image)
{
	Corners corners;

	cv::Mat imageGray = image;
	if (image.type() != CV_8U)
	{
		cv::cvtColor(image, imageGray, CV_RGB2GRAY);
	}

	cv::goodFeaturesToTrack(imageGray, corners, maxCorners_,
							params_.qualityLevel, params_.minDistance, mask_,
							params_.blockSize, true);

	return corners;
}

void FeatureDetector::updatePatches(const common::EventSample& event)
{
	for (auto& patch : patches_)
	{
		if (!patch.isLost())
		{
			if (patch.isInPatch(event.value.point))
			{
				patch.addEvent(event);
			}

			if (event.timestamp - patch.getTimeLastUpdate() >
				patch.getTimeWithoutUpdate())
			{
				consoleLog_->info(
					"Lost patch " + std::to_string(patch.getTrackId()) +
					" because timeWithoutUpdate has been "
					"reached:\ntime since last update: " +
					std::to_string(
						(event.timestamp - patch.getTimeLastUpdate()).count()) +
					" microseconds vs timeWithoutUpdate: " +
					std::to_string(patch.getTimeWithoutUpdate().count()) +
					" microseconds.");
				patch.setLost();
			}

			if (patch.isReady() && patch.isInit() && !patch.isLost())
			{
				optimizer_->optimize(patch);
				updateNumOfEvents(patch);

				if (params_.drawImages)
				{
					patch.warpImage(gradX_, gradY_);
				}
			}
		}
	}
}

void FeatureDetector::addEvent(const common::EventSample& event)
{
	lastEvents_.push_back(event);
	while (lastEvents_.size() > params_.maxNumEventsToStore)
	{
		lastEvents_.pop_front();
	}
}

void FeatureDetector::associatePatches(Patches& newPatches,
									   const common::timestamp_t& timestamp)
{
	// greedy association...
	for (auto& patch : patches_)
	{
		const auto& corner = patch.toCorner();
		for (auto& newPatch : newPatches)
		{
			const auto& newCorner = newPatch.toCorner();
			if (newPatch.getTrackId() == -1 &&
				cv::norm(corner - newCorner) < params_.associationDistance)
			{
				// maybe update respective corner
				newPatch.setTrackId(patch.getTrackId());
				patch.setCorner(newPatch.toCorner(), timestamp);
				break;
			}
		}
	}

	for (auto& newPatch : newPatches)
	{
		if (newPatch.getTrackId() == -1)
		{
			newPatch.setTrackId(nextTrackId_);
			patches_.push_back(newPatch);
			nextTrackId_++;
		}
	}
}

void FeatureDetector::updateNumOfEvents(Patch& patch)
{
	const auto rect = patch.getPatch();
	const auto center = (rect.tl() + rect.br()) * 0.5;
	if (center.x <= 5 || center.y <= 5 || center.x >= gradX_.cols - 5 ||
		center.y >= gradX_.rows - 5)
	{
		consoleLog_->debug("Lost patch number " +
						   std::to_string(patch.getTrackId()));
		patch.setLost();
		return;
	}

	if (rect.x < 0 || rect.y < 0 || rect.x + rect.width >= gradX_.cols ||
		rect.y + rect.height >= gradX_.rows)
	{
		patch.setNumOfEvents(params_.initNumEvents);
		consoleLog_->debug(std::to_string(patch.getNumOfEvents()) +
						   " events are required for patch in track " +
						   std::to_string(patch.getTrackId()));
		return;
	}

	cv::Mat warpedGradX;
	cv::Mat warpedGradY;

	cv::Mat warpCv;
	cv::eigen2cv(patch.getWarp().matrix2x3(), warpCv);

	cv::warpAffine(gradX_, warpedGradX, warpCv, {gradX_.cols, gradX_.rows},
				   cv::WARP_INVERSE_MAP);
	cv::warpAffine(gradY_, warpedGradY, warpCv, {gradY_.cols, gradY_.rows},
				   cv::WARP_INVERSE_MAP);

	const auto gradX = warpedGradX(rect);
	const auto gradY = warpedGradY(rect);
	const auto flow = patch.getFlow();
	size_t sumPatch =
		cv::norm(gradX * std::cos(flow) + gradY * std::sin(flow), cv::NORM_L1);

	patch.setNumOfEvents(sumPatch);

	consoleLog_->debug(std::to_string(patch.getNumOfEvents()) +
					   " events are required for patch in track " +
					   std::to_string(patch.getTrackId()));
}

cv::Mat FeatureDetector::getLogImage(const cv::Mat& image)
{
	assert(image.type() == CV_8U);
	cv::Mat normalizedImage;
	cv::Mat logImage;

	image.convertTo(normalizedImage, CV_64F, 1.0 / 255.0);
	cv::log(normalizedImage + 10e-5, logImage);
	return logImage;
}

cv::Mat FeatureDetector::getGradients(const cv::Mat& image, bool xDir)
{
	assert(image.type() == CV_64F);

	cv::Mat grad;
	cv::Sobel(image / 8, grad, CV_64F, xDir, !xDir, 3);
	return grad;
}

void FeatureDetector::setParams(const tracker::DetectorParams& params)
{
	params_ = params;
	optimizer_->setParams(params_.optimizerParams);
	reset();
}

std::vector<tracker::OptimizerFinalLoss>
FeatureDetector::getOptimizedFinalCosts()
{
	return optimizer_->getFinalCosts();
}

cv::Mat const& FeatureDetector::getCompensatedEventImage()
{
	return compensatedEventImage_;
}

cv::Mat const& FeatureDetector::getIntegratedEventImage()
{
	return integratedEventImage_;
}

common::timestamp_t const& FeatureDetector::getLastCompensation()
{
	return lastCompensation;
}

}  // namespace tracker