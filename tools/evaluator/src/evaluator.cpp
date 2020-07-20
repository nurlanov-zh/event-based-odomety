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
	try
	{
		tracker_->preExit();
	}
	catch (const std::exception&)
	{
		errLog_->warn("Unable to pre exit VO");
	}

	try
	{
		visualOdometry_->preExit();
	}
	catch (const std::exception&)
	{
		errLog_->warn("Unable to pre exit VO");
	}

	try
	{
		saveFeaturesTrajectory(tracker_->getArchivedPatches());
	}
	catch (const std::exception&)
	{
		errLog_->warn("Unable to save trajectory");
	}

	try
	{
		savePoses(visualOdometry_->getAlignedFrames());
	}
	catch (const std::exception&)
	{
		errLog_->warn("Unable to save poses");
	}

	try
	{
		saveGt(visualOdometry_->getGtPoses());
	}
	catch (const std::exception&)
	{
		errLog_->warn("Unable to save gt");
	}

	try
	{
		saveMap(visualOdometry_->getStoredLandmarks());
	}
	catch (const std::exception&)
	{
		errLog_->warn("Unable to save map");
	}

	try
	{
		saveFinalCosts(tracker_->getOptimizedFinalCosts());
	}
	catch (const std::exception&)
	{
		errLog_->warn("Unable to save final cost");
	}
}

tracker::Patches const& Evaluator::getPatches() const
{
	if (params_.visOdometryExperiment)
	{
		return patches_;
	}
	return tracker_->getPatches();
}

void Evaluator::eventCallback(const common::EventSample& sample)
{
	tracker_->addEvent(sample);
	tracker_->updatePatches(sample);
	if ((sample.timestamp - tracker_->getLastCompensation()).count() >=
			params_.compensationFrequencyTime &&
		tracker_->getEvents().size() >= params_.compensationFrequencyEvents)
	{
		//		 tracker_->compensateEvents(tracker_->getEvents());
	}
}

void Evaluator::groundTruthCallback(const common::GroundTruthSample& /*sample*/)
{
}

void Evaluator::imageCallback(const common::ImageSample& sample)
{
	if (!tracker_->isReadyToCompensate() && imageNum_ < 1)
	{
		consoleLog_->info("Should be image at timestamp " +
						  std::to_string(sample.timestamp.count()) +
						  ".\nBut there are not enough events to compensate!");
		return;
	}
	consoleLog_->info("New image at timestamp " +
					  std::to_string(sample.timestamp.count()));
	consoleLog_->info("Using compensated event image instead!");

	tracker_->compensateEventsContrast(tracker_->getEvents());
	tracker_->integrateEvents(tracker_->getEvents());
	//	tracker_->clearEvents();
	cv::Mat image = tracker_->getCompensatedEventImage();

	common::ImageSample imageSample;
	imageSample.timestamp = sample.timestamp;
	imageSample.value = image;
	//	imageCallback(imageSample);

	imageNum_++;
	if (params_.trackerExperiment)
	{
		if (imageNum_ > 2)
		{
			return;
		}
	}

	if (!params_.visOdometryExperiment)
	{
		tracker_->newImage(imageSample);
		corners_ = tracker_->getFeatures();
	}
	else
	{
		const auto it = keyframes_.find(imageSample.timestamp.count());
		if (it != keyframes_.end())
		{
			patches_ = it->second;
		}
	}

	if (imageNum_ > 2)
	{
		if (!params_.visOdometryExperiment)
		{
			auto keyframe = visual_odometry::Keyframe(tracker_->getPatches(),
													  imageSample.timestamp);
			visualOdometry_->newKeyframeCandidate(keyframe);
		}
		else
		{
			std::cout << "here" << std::endl;
			const auto it = keyframes_.find(imageSample.timestamp.count());
			if (it != keyframes_.end())
			{
				auto keyframe = visual_odometry::Keyframe(
					it->second, imageSample.timestamp);
				visualOdometry_->newKeyframeCandidate(keyframe);
			}
		}
	}
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

void Evaluator::saveFeaturesTrajectory(const tracker::Patches& patches)
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
					 /*+ 1457432569.609347070 -
					 std::chrono::duration<double>(
						patch.getTrajectory().front().timestamp)
						.count()*/
					 << " " << pos.value.x << " " << pos.value.y << std::endl;
		}
	}
	trajFile.close();
	consoleLog_->info("Saved!");
}

void Evaluator::saveMap(
	const std::vector<std::pair<tracker::TrackId, Eigen::Vector3d>>& map)
{
	const std::string outputFilename = params_.outputDir + "/map.txt";
	consoleLog_->info("Saving map into " + outputFilename);
	std::ofstream mapFile;
	mapFile.open(outputFilename);
	for (const auto& lm : map)
	{
		mapFile << std::fixed << std::setprecision(8) << lm.first << " "
				<< lm.second(0) << " " << lm.second(1) << " " << lm.second(2)
				<< std::endl;
	}
	mapFile.close();
	consoleLog_->info("Saved!");
}

void Evaluator::savePoses(const std::list<visual_odometry::Keyframe>& keyframes)
{
	const std::string outputFilename = params_.outputDir + "/vo_trajectory.txt";
	consoleLog_->info("Saving VO trajectories into " + outputFilename);
	std::ofstream trajFile;
	trajFile.open(outputFilename);
	for (const auto& kf : keyframes)
	{
		// feature_id timestamp x y
		trajFile << std::fixed << std::setprecision(8)
				 << kf.pose.matrix3x4()(0, 0) << " "
				 << kf.pose.matrix3x4()(0, 1) << " "
				 << kf.pose.matrix3x4()(0, 2) << " "
				 << kf.pose.matrix3x4()(0, 3) << " "
				 << kf.pose.matrix3x4()(1, 0) << " "
				 << kf.pose.matrix3x4()(1, 1) << " "
				 << kf.pose.matrix3x4()(1, 2) << " "
				 << kf.pose.matrix3x4()(1, 3) << " "
				 << kf.pose.matrix3x4()(2, 0) << " "
				 << kf.pose.matrix3x4()(2, 1) << " "
				 << kf.pose.matrix3x4()(2, 2) << " "
				 << kf.pose.matrix3x4()(2, 3) << std::endl;
	}
	trajFile.close();
	consoleLog_->info("Saved!");
}

void Evaluator::saveGt(const std::vector<common::Pose3d>& gts)
{
	const std::string outputFilename =
		params_.outputDir + "/groundtruth_aligned.txt";
	consoleLog_->info("Saving GT trajectories into " + outputFilename);
	std::ofstream trajFile;
	trajFile.open(outputFilename);
	for (const auto& gt : gts)
	{
		// feature_id timestamp x y
		trajFile << std::fixed << std::setprecision(8) << gt.matrix3x4()(0, 0)
				 << " " << gt.matrix3x4()(0, 1) << " " << gt.matrix3x4()(0, 2)
				 << " " << gt.matrix3x4()(0, 3) << " " << gt.matrix3x4()(1, 0)
				 << " " << gt.matrix3x4()(1, 1) << " " << gt.matrix3x4()(1, 2)
				 << " " << gt.matrix3x4()(1, 3) << " " << gt.matrix3x4()(2, 0)
				 << " " << gt.matrix3x4()(2, 1) << " " << gt.matrix3x4()(2, 2)
				 << " " << gt.matrix3x4()(2, 3) << std::endl;
	}
	trajFile.close();
	consoleLog_->info("Saved!");
}

void Evaluator::setGroundTruthSamples(
	const common::GroundTruth& groundTruthSamples)
{
	visualOdometry_->setGroundTruthSamples(groundTruthSamples);
}

void Evaluator::setPatches(const tracker::Patches& patches)
{
	for (const auto& patch : patches)
	{
		keyframes_[patch.getCurrentTimestamp().count()].push_back(patch);
	}
}

visual_odometry::MapLandmarks const& Evaluator::getMapLandmarks()
{
	return visualOdometry_->getMapLandmarks();
}

std::map<size_t, visual_odometry::Keyframe> const& Evaluator::getActiveFrames()
	const
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

std::vector<common::Pose3d> const& Evaluator::getGtPoses() const
{
	return visualOdometry_->getGtAlignedPoses();
}

std::vector<std::pair<tracker::TrackId, Eigen::Vector3d>> const&
Evaluator::getStoredMapLandmarks() const
{
	return visualOdometry_->getStoredLandmarks();
}
}  // namespace tools
