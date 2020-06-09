#include "visual_odometry/triangulation.h"

#include <opengv/triangulation/methods.hpp>

namespace visual_odometry
{
// void findCommonTrajectories
std::vector<Eigen::Vector3d> triangulateLandmarks(
	const common::Pose3d& cam1Pose, const common::Pose3d& cam2Pose,
	const opengv::bearingVectors_t& bearingVectors1,
	const opengv::bearingVectors_t& bearingVectors2)
{
	assert(bearingVectors1.size() == bearingVectors2.size());
	const auto cam1To2Pose = cam1Pose.inverse() * cam2Pose;

	const opengv::translation_t translation = cam1To2Pose.translation();
	const opengv::rotation_t rotation = cam1To2Pose.rotationMatrix();

	opengv::relative_pose::CentralRelativeAdapter adapter(
		bearingVectors1, bearingVectors2, translation, rotation);

	std::vector<Eigen::Vector3d> triangulatedLandmarks(bearingVectors1.size());

	for (size_t i = 0; i < bearingVectors1.size(); ++i)
	{
		triangulatedLandmarks[i] =
			cam1Pose * opengv::triangulation::triangulate2(adapter, i);
	}
	return triangulatedLandmarks;
}

std::vector<common::Point2d, common::Point2d> getSharedLandmarks(
	const common::timestamp_t& time1, const common::timestamp_t& time2)
{
	// keyframes_[time1]
}

}  // namespace visual_odometry