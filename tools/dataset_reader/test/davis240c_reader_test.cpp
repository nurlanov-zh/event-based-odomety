#include <dataset_reader/davis240c_reader.h>

#include <gtest/gtest.h>

std::string TEST_DATA_PATH = "test/test_data";

class LoggingTest
{
   public:
	LoggingTest()
	{
		spdlog::stdout_color_mt("console");
		spdlog::stderr_color_mt("stderr");
	}
};

LoggingTest logger;

TEST(Davis240cReader, eventsTest)
{
	tools::Davis240cReader reader(TEST_DATA_PATH);
	const auto events = reader.getEvents();

	std::vector<common::Point2i> points = {
		{33, 39}, {158, 145}, {88, 143}, {174, 154}, {112, 139}};

	std::vector<common::EventPolarity> signs = {
		common::EventPolarity::POSITIVE, common::EventPolarity::POSITIVE,
		common::EventPolarity::NEGATIVE, common::EventPolarity::NEGATIVE,
		common::EventPolarity::POSITIVE};

	std::vector<common::timestamp_t> timestamps = {
		common::timestamp_t(0), common::timestamp_t(11),
		common::timestamp_t(50), common::timestamp_t(55),
		common::timestamp_t(80)};

	ASSERT_EQ(timestamps.size(), events.size());

	for (size_t i = 0; i < events.size(); ++i)
	{
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

	ASSERT_EQ(timestamps.size(), images.size());

	for (size_t i = 0; i < images.size(); ++i)
	{
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
		mat(0, 0) = 1;
		mat(1, 2) = -1;
		mat(2, 1) = 1;
		transforms.push_back({mat, {1, 0, 0}});
	}
	{
		Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
		mat(0, 2) = 1;
		mat(1, 1) = 1;
		mat(2, 0) = -1;
		transforms.push_back({mat, {0, 0, 1}});
	}

	ASSERT_EQ(timestamps.size(), groundTruth.size());

	for (size_t i = 0; i < groundTruth.size(); ++i)
	{
		EXPECT_TRUE(
			groundTruth[i].value.matrix().isApprox(transforms[i].matrix()));
		EXPECT_EQ(groundTruth[i].timestamp.count(), timestamps[i].count());
	}
}

TEST(Davis240cReader, calibrationTest)
{
	tools::Davis240cReader reader(TEST_DATA_PATH);
	const auto calibration = reader.getCalibration();

	EXPECT_FLOAT_EQ(calibration.fx, 501);
	EXPECT_FLOAT_EQ(calibration.fy, 499);
	EXPECT_FLOAT_EQ(calibration.cx, 249);
	EXPECT_FLOAT_EQ(calibration.cy, 251);
	EXPECT_FLOAT_EQ(calibration.k1, 0.11);
	EXPECT_FLOAT_EQ(calibration.k2, 0.011);
	EXPECT_FLOAT_EQ(calibration.k3, 0.0011);
	EXPECT_FLOAT_EQ(calibration.p1, 0.123);
	EXPECT_FLOAT_EQ(calibration.p2, 0.321);
}