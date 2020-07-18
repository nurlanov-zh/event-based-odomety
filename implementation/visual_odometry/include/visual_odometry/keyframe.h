#pragma once

#include <common/data_types.h>
#include <feature_tracker/patch.h>

#include <vector>

namespace visual_odometry
{
using Landmarks = std::unordered_map<tracker::TrackId, Eigen::Vector2d>;

struct Match
{
	common::Pose3d Tw2c;
	std::vector<tracker::TrackId> inliers;
};

struct MapLandmarks
{
	std::unordered_map<tracker::TrackId, Eigen::Vector3d> landmarks;
	std::unordered_map<tracker::TrackId, std::list<size_t>> observations;
};

class Keyframe
{
   public:
	Keyframe() {}
	Keyframe(const tracker::Patches& patches,
			 const common::timestamp_t& timestamp);

	const Landmarks& getLandmarks() const { return landmarks_; }

	std::vector<tracker::TrackId> getSharedTracks(const Keyframe& frame) const;

   public:
	common::Pose3d pose;
	common::timestamp_t timestamp;

   private:
	Landmarks landmarks_;
};
}  // namespace visual_odometry