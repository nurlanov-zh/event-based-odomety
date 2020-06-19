#include "visual_odometry/triangulation.h"

#include <opengv/triangulation/methods.hpp>

namespace visual_odometry
{
std::vector<Eigen::Vector3d> triangulateLandmarks(
	const common::Pose3d& cam1Pose, const common::Pose3d& cam2Pose,
	const opengv::bearingVectors_t& bearingVectors1,
	const opengv::bearingVectors_t& bearingVectors2)
{
	assert(bearingVectors1.size() == bearingVectors2.size());
	const auto cam2To1Pose = cam1Pose.inverse() * cam2Pose;

	const opengv::translation_t translation = cam2To1Pose.translation();
	const opengv::rotation_t rotation = cam2To1Pose.rotationMatrix();

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

}  // namespace visual_odometry