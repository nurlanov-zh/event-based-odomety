#include "visual_odometry/visual_odometry.h"
#include "visual_odometry/triangulation.h"

#include <sophus/interpolate.hpp>

#include <cmath>

namespace visual_odometry
{
VisualOdometryFrontEnd::VisualOdometryFrontEnd(
	const common::CameraModelParams& calibration)
{
	cameraModel_.reset(new common::CameraModel<double>(calibration));

	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");
}

void VisualOdometryFrontEnd::newKeyframeCandidate(const Keyframe& keyframe)
{
	for (auto& frame : activeFrames_)
	{
		const auto firstImagePose = syncGtAndImage(frame.timestamp);
		const auto secondImagePose = syncGtAndImage(keyframe.timestamp);

		if (!firstImagePose.has_value() || !secondImagePose.has_value())
		{
			continue;
		}

		frame.pose = firstImagePose.value();

		if ((firstImagePose.value().translation() -
			 secondImagePose.value().translation())
				.norm() < 0.1)
		{
			continue;
		}

		opengv::bearingVectors_t bearingVectors1;
		opengv::bearingVectors_t bearingVectors2;
		std::vector<tracker::TrackId> trackIds;

		getCommonBearingVectors(frame, keyframe, trackIds, bearingVectors1,
								bearingVectors2);

		const auto landmarks = triangulateLandmarks(
			firstImagePose.value(), secondImagePose.value(), bearingVectors1,
			bearingVectors2);

		for (size_t i = 0; i < trackIds.size(); ++i)
		{
			if (!std::isnan(landmarks[i].x()) &&
				!std::isnan(landmarks[i].y()) && !std::isnan(landmarks[i].z()))
			{
				mapLandmarks_[trackIds[i]] = landmarks[i];
			}
		}
	}

	activeFrames_.push_back(keyframe);

	if (activeFrames_.size() > 10)
	{
		storedFrames_.push_back(activeFrames_.front());
		activeFrames_.pop_front();
	}

	consoleLog_->info("Map consist of " + std::to_string(mapLandmarks_.size()) +
					  " landmarks");
}

void VisualOdometryFrontEnd::getCommonBearingVectors(
	const Keyframe& keyframe1, const Keyframe& keyframe2,
	std::vector<tracker::TrackId>& trackIds,
	opengv::bearingVectors_t& bearingVectors1,
	opengv::bearingVectors_t& bearingVectors2)
{
	const auto& sharedTracks = keyframe1.getSharedTracks(keyframe2);
	const auto& landmarks1 = keyframe1.getLandmarks();
	const auto& landmarks2 = keyframe2.getLandmarks();

	for (const auto& track : sharedTracks)
	{
		// const auto landmarkIt = mapLandmarks_.find(track);
		// if (landmarkIt == mapLandmarks_.end())
		{
			const auto landmarks1It = landmarks1.find(track);
			const auto landmarks2It = landmarks2.find(track);
			trackIds.push_back(track);

			bearingVectors1.push_back(
				cameraModel_->unproject(landmarks1It->second));
			bearingVectors2.push_back(
				cameraModel_->unproject(landmarks2It->second));
		}
	}
}

std::optional<common::Pose3d> VisualOdometryFrontEnd::syncGtAndImage(
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
	const auto nextPose = *lowerBoundIt;

	if (lowerBoundIt->timestamp == timestamp)
	{
		return std::make_optional(lowerBoundIt->value);
	}

	if (lowerBoundIt == groundTruthSamples_.begin())
	{
		return {};
	}

	--lowerBoundIt;
	if (lowerBoundIt == groundTruthSamples_.end())
	{
		return {};
	}
	const auto prevPose = *lowerBoundIt;

	const auto interploatedPose = Sophus::interpolate(
		prevPose.value, nextPose.value,
		static_cast<float>((timestamp - prevPose.timestamp).count()) /
			(nextPose.timestamp - prevPose.timestamp).count());

	return std::make_optional(interploatedPose);
}

void VisualOdometryFrontEnd::setGroundTruthSamples(
	const common::GroundTruth& groundTruthSamples)
{
	groundTruthSamples_ = groundTruthSamples;
}

MapLandmarks const& VisualOdometryFrontEnd::getMapLandmarks()
{
	return mapLandmarks_;
}

std::list<Keyframe> const& VisualOdometryFrontEnd::getActiveFrames() const
{
	return activeFrames_;
}

std::list<Keyframe> const& VisualOdometryFrontEnd::getStoredFrames() const
{
	return storedFrames_;
}

}  // namespace visual_odometry