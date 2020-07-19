#pragma once

#include <common/data_types.h>
#include "visual_odometry/keyframe.h"

namespace visual_odometry
{
struct ErrorMetricValue
{
	double rmse = 0;
	double mean = 0;
	double min = 0;
	double max = 0;
	double count = 0;  //!< number of elements involved in the evaluation
};

Sophus::Sim3d align_cameras_sim3(
	const std::vector<common::Pose3d> &reference_poses,
	const std::list<Keyframe> &cameras, ErrorMetricValue *ate);
}  // namespace visual_odometry