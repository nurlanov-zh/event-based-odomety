#include "evaluator/evaluator.h"

#include <visual_odometry/triangulation.h>

#include <sophus/interpolate.hpp>

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
	consoleLog_->info("New image at timestamp " +
					  std::to_string(sample.timestamp.count()));
	imageNum_++;
	if (params_.experiment)
	{
		if (imageNum_ > 2)
		{
			return;
		}
	}

	tracker_->newImage(sample);
	corners_ = tracker_->getFeatures();
	
	imageTimestamps_.push_back(sample.timestamp);

	/**********************************/
	if (imageTimestamps_.size() < 2)
	{
		return;
	}

	const auto firstImagePose = syncGtAndImage(imageTimestamps_[0]);
	const auto secondImagePose = syncGtAndImage(imageTimestamps_[1]);

	if (!firstImagePose.has_value() || !secondImagePose.has_value())
	{
		return;
	}

	// const auto& sharedLandmarks = vo::getSharedLandmarks(imageTimestamps_[0],
	// imageTimestamps_[1]);
	vo::triangulateLandmarks(firstImagePose.value(), secondImagePose.value(),
							 {}, {});

	imageTimestamps_.erase(imageTimestamps_.begin());
}

std::optional<common::Pose3d> Evaluator::syncGtAndImage(
	const common::timestamp_t& timestamp)
{
	auto lowerBoundIt =
		std::lower_bound(groundTruthSamples_.begin(), groundTruthSamples_.end(),
						 common::GroundTruthSample({}, timestamp),
						 [](const common::GroundTruthSample& a,
							const common::GroundTruthSample& b) {
							 return a.timestamp < b.timestamp;
						 });
	if (lowerBoundIt == groundTruthSamples_.end())
	{
		return {};
	}
	const auto prevPose = lowerBoundIt->value;

	auto nextIt = (++lowerBoundIt);
	if (nextIt == groundTruthSamples_.end())
	{
		return {};
	}
	const auto nextPose = nextIt->value;

	const auto interploatedPose = Sophus::interpolate(
		prevPose, nextPose,
		static_cast<float>((timestamp - lowerBoundIt->timestamp).count()) /
			(nextIt->timestamp - lowerBoundIt->timestamp).count());

	// auto itErase = groundTruthSamples_.begin();
	// while (nextIt != itErase)
	// {
	// 	itErase = groundTruthSamples_.erase(itErase);
	// }
	return std::make_optional(interploatedPose);
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
}  // namespace tools
