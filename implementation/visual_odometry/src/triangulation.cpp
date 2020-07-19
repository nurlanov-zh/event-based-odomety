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

Eigen::Matrix3d computeEssential(const Sophus::SE3d& T_0_1)
{
	const Eigen::Vector3d t_0_1 = T_0_1.translation();
	const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

	return Sophus::SO3d::hat(t_0_1.normalized()) * R_0_1;
}

void findInliersEssential(const opengv::bearingVectors_t& bearingVectors1,
						  const opengv::bearingVectors_t& bearingVectors2,
						  const Keyframe& keyframe1, const Keyframe& keyframe2,
						  const std::vector<tracker::TrackId>& tracks,
						  Match& match,
						  double epipolarErrorThreshold)
{
	assert(bearingVectors1.size() == bearingVectors2.size());

	const Sophus::SE3d T_1_2 = keyframe2.pose.inverse() * keyframe1.pose;
	const Eigen::Matrix3d& E = computeEssential(T_1_2);

	match.inliers.clear();

	for (size_t j = 0; j < bearingVectors1.size(); j++)
	{
		const double constraint =
			bearingVectors1[j].transpose() * E * bearingVectors2[j];
		if (std::abs(constraint) < epipolarErrorThreshold)
		{
			std::cout << std::abs(constraint) << std::endl;
			match.inliers.push_back(tracks[j]);
		}
	}
}

}  // namespace visual_odometry