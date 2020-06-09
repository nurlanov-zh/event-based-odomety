#pragma once

#include <common/data_types.h>

#include <vector>

namespace visual_odometry
{
struct Landmark
{
	common::Point2d point2d;
	common::Point3d point3d;
};

using Landmarks = std::unordered_map<size_t, Landmark>;

class Keyframe
{
   public:
	Keyframe();

	const Landmarks& getLandmarks() const { return landmarks_; }

	std::vector<size_t> getSharedTracks(Keyframe& frame) const;

	//const std::vector<Landmark>& getSharedLandmarks(Keyframe& frame) const;

   public:
	common::Pose3d pose;
	common::timestamp_t timestamp;

   private:
	Landmarks landmarks_;
};
}  // namespace visual_odometry