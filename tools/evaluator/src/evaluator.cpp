#include "evaluator/evaluator.h"

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
}

tracker::Patches const& Evaluator::getPatches() const
{
	return tracker_->getPatches();
}

void Evaluator::eventCallback(const common::EventSample& sample)
{
	tracker_->updatePatches(sample);
}

void Evaluator::groundTruthCallback(const common::GroundTruthSample& /*sample*/)
{
}

void Evaluator::imageCallback(const common::ImageSample& sample)
{
	imageNum_++;
	if (params_.experiment)
	{
		if (imageNum_ > 2)
		{
			return;
		}
	}

	consoleLog_->info("New image at timestamp " +
					  std::to_string(sample.timestamp.count()));

	tracker_->newImage(sample);
	corners_ = tracker_->getFeatures();

	visualOdometry_->newKeyframeCandidate(
		visual_odometry::Keyframe(tracker_->getPatches(), sample.timestamp));
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
	visualOdometry_.reset(
		new visual_odometry::VisualOdometryFrontEnd(params_.cameraModelParams));
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
					 << " " << pos.value.x << " " << pos.value.y << std::endl;
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
}  // namespace tools
