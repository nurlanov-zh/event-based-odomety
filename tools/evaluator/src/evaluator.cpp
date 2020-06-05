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
	for (auto& patch : tracker_->getPatches()) {
		if (patch.isReady()) {
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

	for (auto& patch : tracker_->getPatches()) {
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

void Evaluator::saveTrajectory(const tracker::Patches& patches)
{
	// Since we are going to use evaluator by uzh-rpg lab. We just need to store
	// trajectory in proper format
	const std::string outputFilename = params_.outputDir + "/trajectory.txt";
	consoleLog_->info("Saving trajectory into " + outputFilename);
	std::ofstream trajFile;
	trajFile.open(outputFilename);
	for (const auto& patch : patches) {
		for (const auto& pos : patch.getTrajectory()) {
			// feature_id timestamp x y
			trajFile << std::fixed << std::setprecision(8) << patch.getTrackId()
					 << " "
					 << std::chrono::duration<double>(pos.timestamp).count()
					 << " " << pos.value.translation().x() << " "
					 << pos.value.translation().y() << std::endl;
		}
	}
	trajFile.close();
	consoleLog_->info("Saved!");
}
}  // ns tools