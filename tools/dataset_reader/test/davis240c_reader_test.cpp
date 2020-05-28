#include <dataset_reader/davis240c_reader.h>

#include <gtest/gtest.h>

std::string TEST_DATA_PATH = "../../../../tools/dataset_reader/test/data";

TEST(Davis240cReader, eventsTest)
{
	tools::Davis240cReader reader(TEST_DATA_PATH);
	const auto events = reader.getEvents();

	std::vector<common::Point2i> points = {
		{33, 39}, {158, 145}, {88, 143}, {174, 154}, {112, 139}};

	std::vector<int8_t> signs = {1, 1, 0, 0, 1};

	std::vector<common::timestamp_t> timestamps = {
		common::timestamp_t(0), common::timestamp_t(11),
		common::timestamp_t(50), common::timestamp_t(55),
		common::timestamp_t(80)};

	for (size_t i = 0; i < events.size(); ++i) {
		EXPECT_EQ(events[i].timestamp.count(), timestamps[i].count());
		EXPECT_EQ(events[i].value.point.x, points[i].x);
		EXPECT_EQ(events[i].value.point.y, points[i].y);
		EXPECT_EQ(events[i].value.sign, signs[i]);
	}
}

TEST(Davis240cReader, imagesTest)
{
	tools::Davis240cReader reader(TEST_DATA_PATH);
	const auto images = reader.getImages();

	std::vector<common::timestamp_t> timestamps = {common::timestamp_t(28046),
												   common::timestamp_t(72111),
												   common::timestamp_t(116176)};

	std::vector<cv::Mat> imagesGT = {
		cv::imread(TEST_DATA_PATH + "/images/frame_00000000.png", CV_8U),
		cv::imread(TEST_DATA_PATH + "/images/frame_00000001.png", CV_8U),
		cv::imread(TEST_DATA_PATH + "/images/frame_00000002.png", CV_8U)};

	for (size_t i = 0; i < images.size(); ++i) {
		const cv::Mat diff = images[i].value != imagesGT[i];
		EXPECT_EQ(cv::countNonZero(diff), 0);
		EXPECT_EQ(images[i].timestamp.count(), timestamps[i].count());
	}
}

TEST(Davis240cReader, groundTruthTest)
{
	tools::Davis240cReader reader(TEST_DATA_PATH);
	const auto groundTruth = reader.getGroundTruth();

	std::vector<common::timestamp_t> timestamps = {common::timestamp_t(72111),
												   common::timestamp_t(116176)};

	std::vector<Sophus::SE3d::Point> points = {{1, 0, 0}, {0, 0, 1}};

	std::vector<Sophus::SE3d> transforms;
	{
		Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
		mat(0, 0)			= 1;
		mat(1, 2)			= -1;
		mat(2, 1)			= 1;
		transforms.push_back({mat, {1, 0, 0}});
	}
	{
		Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
		mat(0, 2)			= 1;
		mat(1, 1)			= 1;
		mat(2, 0)			= -1;
		transforms.push_back({mat, {0, 0, 1}});
	}

	for (size_t i = 0; i < groundTruth.size(); ++i) {
		EXPECT_TRUE(
			groundTruth[i].value.matrix().isApprox(transforms[i].matrix()));
		EXPECT_EQ(groundTruth[i].timestamp.count(), timestamps[i].count());
	}
}