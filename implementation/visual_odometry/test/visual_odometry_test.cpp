#include <visual_odometry/visual_odometry.h>

#include <gtest/gtest.h>

TEST(VisualOdometryTest, syncGtAndImageTest)
{
	common::GroundTruth groundTruthSequence;

	for (int i = 0; i < 3; ++i)
	{
		common::GroundTruthSample gt;
		gt.timestamp = common::timestamp_t(i * 10);
		gt.value.translation() = Eigen::Vector3d(i * 10, 0, 0);
		groundTruthSequence.push_back(gt);
	}

	const auto param = common::CameraModelParams();
	visual_odometry::VisualOdometryFrontEnd visualOdometry(param);

	visualOdometry.setGroundTruthSamples(groundTruthSequence);

	const auto syncPoseEvaluator = [&visualOdometry](int value,
												bool hasValue) -> void {
		const auto pose = visualOdometry.syncGtAndImage(common::timestamp_t(value));
        std::cout << value << std::endl;
		ASSERT_EQ(pose.has_value(), hasValue);
		if (!hasValue)
		{
			return;
		}

		EXPECT_FLOAT_EQ(pose.value().translation().x(), value);
		EXPECT_FLOAT_EQ(pose.value().translation().y(), 0);
		EXPECT_FLOAT_EQ(pose.value().translation().z(), 0);
	};

	syncPoseEvaluator(0, true);
	syncPoseEvaluator(5, true);
	syncPoseEvaluator(25, false);
}