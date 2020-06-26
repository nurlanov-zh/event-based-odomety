#pragma once

#include <common/data_types.h>
#include <feature_tracker/patch.h>

#include <vector>

namespace visual_odometry
{
using MapLandmarks = std::unordered_map<tracker::TrackId, Eigen::Vector3d>;
using Landmarks = std::unordered_map<tracker::TrackId, Eigen::Vector2d>;

struct Match
{
	common::Pose3d T_i_j;
	std::vector<tracker::TrackId> inliers;
};

class Keyframe
{
   public:
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