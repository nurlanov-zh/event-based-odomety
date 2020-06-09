#include <visual_odometry/triangulation.h>

#include <gtest/gtest.h>

TEST(TriangulationTest, simpleCase)
{
	common::Pose3d camera1Pose;
	opengv::bearingVectors_t bearingVector1 = {{1.0, 0.0, 0.0}};

	const Eigen::Matrix3d R =
		Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1))
			.toRotationMatrix();
	const Eigen::Vector3d t = Eigen::Vector3d(1.0, -1.0, 0.0);
	common::Pose3d camera2Pose(R, t);
	opengv::bearingVectors_t bearingVector2 = {{1.0, 0.0, 0.0}};

	const auto& triangulatedLandmarks = visual_odometry::triangulateLandmarks(
		camera1Pose, camera2Pose, bearingVector1, bearingVector2);

	EXPECT_FLOAT_EQ(triangulatedLandmarks[0].x(), 1);
	EXPECT_FLOAT_EQ(triangulatedLandmarks[0].y(), 0);
	EXPECT_FLOAT_EQ(triangulatedLandmarks[0].z(), 0);
}