#include "evaluator/evaluator.h"
#include <fstream>

#include <visual_odometry/triangulation.h>

namespace vo = visual_odometry;

namespace tools
{
Evaluator::Evaluator(const EvaluatorParams& params) : params_(params)
{
	reset();
}

Evaluator::~Evaluator()
{
	tracker_->preExit();
	saveTrajectory(tracker_->getArchivedPatches());
	saveFinalCosts(tracker_->getOptimizedFinalCosts());
}

tracker::Patches const& Evaluator::getPatches() const
{
	return tracker_->getPatches();
}

void Evaluator::eventCallback(const common::EventSample& sample)
{
	tracker_->addEvent(sample);
	tracker_->updatePatches(sample);
	if ((sample.timestamp - tracker_->getLastCompensation()).count() >=
			params_.compensationFrequencyTime or
		tracker_->getEvents().size() >= params_.compensationFrequencyEvents)
	{
		//		tracker_->compensateEvents(tracker_->getEvents());
		tracker_->compensateEventsContrast(tracker_->getEvents());
		tracker_->integrateEvents(tracker_->getEvents());
		tracker_->clearEvents();
	}
}

void Evaluator::groundTruthCallback(const common::GroundTruthSample& /*sample*/)
{
}

void Evaluator::imageCallback(const common::ImageSample& sample)
{
	if (params_.experiment)
	{
		if (imageNum_ > 2)
		{
			return;
		}
	}
	imageNum_++;

	consoleLog_->info("New image at timestamp " +
					  std::to_string(sample.timestamp.count()));

	tracker_->newImage(sample);
	corners_ = tracker_->getFeatures();

	auto keyframe =
		visual_odometry::Keyframe(tracker_->getPatches(), sample.timestamp);
	visualOdometry_->newKeyframeCandidate(keyframe);
}

void Evaluator::reset()
{
	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");

	corners_.clear();
	tracker::DetectorParams params;
	params.drawImages = params_.drawImages;
	params.imageSize = params_.imageSize;
	tracker_.reset(new tracker::FeatureDetector(params));
	visualOdometry_.reset(new visual_odometry::VisualOdometryFrontEnd(
		params_.cameraModelParams, visual_odometry::VisualOdometryParams()));
	imageNum_ = 0;

	consoleLog_->info("Evaluator is reset");
}

void Evaluator::setTrackerParams(const tracker::DetectorParams& params)
{
	tracker_->setParams(params);
}

void Evaluator::saveTrajectory(const tracker::Patches& patches)
{
	// Since we are going to use evaluator by uzh-rpg lab. We just need to store
	// trajectory in proper format
	const std::string outputFilename = params_.outputDir + "/trajectory.txt";
	consoleLog_->info("Saving trajectory into " + outputFilename);
	std::ofstream trajFile;
	trajFile.open(outputFilename);
	for (const auto& patch : patches)
	{
		for (const auto& pos : patch.getTrajectory())
		{
			// feature_id timestamp x y
			trajFile << std::fixed << std::setprecision(8) << patch.getTrackId()
					 << " "
					 << std::chrono::duration<double>(pos.timestamp).count()
					 << " " << round(pos.value.x) << " " << round(pos.value.y)
					 << std::endl;
		}
	}
	trajFile.close();
	consoleLog_->info("Saved!");
}

void Evaluator::setGroundTruthSamples(
	const common::GroundTruth& groundTruthSamples)
{
	visualOdometry_->setGroundTruthSamples(groundTruthSamples);
}

visual_odometry::MapLandmarks const& Evaluator::getMapLandmarks()
{
	return visualOdometry_->getMapLandmarks();
}

std::list<visual_odometry::Keyframe> const& Evaluator::getActiveFrames() const
{
	return visualOdometry_->getActiveFrames();
}

std::list<visual_odometry::Keyframe> const& Evaluator::getStoredFrames() const
{
	return visualOdometry_->getStoredFrames();
}

void Evaluator::saveFinalCosts(
	const std::vector<tracker::OptimizerFinalLoss>& vectorFinalCosts)
{
	const std::string outputFilename = params_.outputDir + "/final_cost.txt";
	consoleLog_->info("Saving final costs into " + outputFilename);
	std::ofstream costFile;
	costFile.open(outputFilename);

	for (const auto& v : vectorFinalCosts)
	{
		costFile << v.trackId << " " << std::fixed << std::setprecision(8)
				 << v.lossValue << " " << v.timeStampMicrosecond << std::endl;
	}

	costFile.close();
	consoleLog_->info("Saved!");
}

cv::Mat const& Evaluator::getCompensatedEventImage()
{
	return tracker_->getCompensatedEventImage();
}

cv::Mat const& Evaluator::getIntegratedEventImage()
{
	return tracker_->getIntegratedEventImage();
}

}  // namespace tools
