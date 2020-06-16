#pragma once

#include <common/data_types.h>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>

#include <vector>

namespace visual_odometry
{
std::vector<Eigen::Vector3d> triangulateLandmarks(
	const common::Pose3d& cam1Pose, const common::Pose3d& cam2Pose,
	const opengv::bearingVectors_t& bearingVectors1,
	const opengv::bearingVectors_t& bearingVectors2);
}  // namespace visual_odometry