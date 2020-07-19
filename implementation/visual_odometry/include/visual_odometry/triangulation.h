#pragma once

#include <common/data_types.h>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include "visual_odometry/keyframe.h"

#include <vector>

namespace visual_odometry
{
std::vector<Eigen::Vector3d> triangulateLandmarks(
	const common::Pose3d& cam1Pose, const common::Pose3d& cam2Pose,
	const opengv::bearingVectors_t& bearingVectors1,
	const opengv::bearingVectors_t& bearingVectors2);

void findInliersEssential(const opengv::bearingVectors_t& bearingVectors1,
						  const opengv::bearingVectors_t& bearingVectors2,
						  const Keyframe& keyframe1, const Keyframe& keyframe2,
						  const std::vector<tracker::TrackId>& tracks,
						  Match& match, double epipolarErrorThreshold);
}  // namespace visual_odometry