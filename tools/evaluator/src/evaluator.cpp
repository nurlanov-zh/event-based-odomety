#include "evaluator/evaluator.h"

namespace tools
{
Evaluator::Evaluator(const EvaluatorParams& params) : params_(params)
{
	reset();
}

tracker::Patches const& Evaluator::getPatches() const
{
	return tracker_->getPatches();
}

void Evaluator::eventCallback(const common::EventSample& sample)
{
	tracker_->updatePatches(sample);

	// TODO add logger trace which patch is integrated as soon as patch ids come
	// up
	for (auto& patch : tracker_->getPatches())
	{
		if (patch.isReady())
		{
			patch.integrateEvents();
			patch.resetBatch();
		}
	}
}

void Evaluator::groundTruthCallback(const common::GroundTruthSample& /*sample*/)
{
}

void Evaluator::imageCallback(const common::ImageSample& sample)
{
	tracker_->extractPatches(sample.value);
	corners_ = tracker_->getFeatures();

	flowEstimator_->addImage(sample.value);
	flowEstimator_->getFlowPatches(tracker_->getPatches());

	for (auto& patch : tracker_->getPatches())
	{
		patch.warpImage();
	}
}

void Evaluator::reset()
{
	consoleLog_ = spdlog::get("console");
	errLog_		= spdlog::get("stderr");

	corners_.clear();
	tracker_.reset(new tracker::FeatureDetector(tracker::DetectorParams(),
												params_.imageSize));
	flowEstimator_.reset(
		new tracker::FlowEstimator(tracker::FlowEstimatorParams()));

	consoleLog_->info("Evaluator is reset");
}
}  // namespace tools