#include <feature_tracker/patch.h>

#include <gtest/gtest.h>

std::string TEST_DATA_PATH = "test/test_data";

TEST(Patch, addEventsTest)
{
	tracker::Patch patch({10, 10}, 5, common::timestamp_t(0));
	patch.setNumOfEvents(30);
	for (size_t i = 0; i < 30; ++i)
	{
		common::EventSample event;
		event.timestamp = common::timestamp_t(i);
		event.value.point = {5 + std::rand() % 10, 5 + std::rand() % 10};
		event.value.sign = std::rand() % 2 == 1
							   ? common::EventPolarity::POSITIVE
							   : common::EventPolarity::NEGATIVE;
		patch.addEvent(event);
	}

	EXPECT_TRUE(patch.isReady());

	const auto& events = patch.getEvents();

	ASSERT_EQ(events.size(), 30);
	EXPECT_EQ(events.front().timestamp.count(), 0);
	EXPECT_EQ(events.back().timestamp.count(), 29);

	patch.resetBatch();

	EXPECT_FALSE(patch.isReady());
}

TEST(Patch, integrateEventsTest)
{
	tracker::Patch patch({10, 10}, 3, common::timestamp_t(0));
	patch.setNumOfEvents(30);

	for (int32_t i = 0; i < 30; ++i)
	{
		common::EventSample event;
		event.timestamp = common::timestamp_t(i);
		event.value.point = {7 + i / 7, 7 + i % 7};
		event.value.sign = i % 2 == 0 ? common::EventPolarity::POSITIVE
									  : common::EventPolarity::NEGATIVE;
		patch.addEvent(event);
	}

	patch.integrateEvents();

	const auto& nabla = patch.getIntegratedNabla();

	for (int32_t i = 0; i < 30; ++i)
	{
		EXPECT_FLOAT_EQ(nabla.at<double>(i % 7, i / 7),
						i % 2 == 0 ? common::EventPolarity::POSITIVE
								   : common::EventPolarity::NEGATIVE);
	}
}

TEST(Patch, warpImageTest)
{
	tracker::Patch patch({5, 5}, 5, common::timestamp_t(0));

	cv::Mat gradX = cv::Mat::zeros(11, 11, CV_64F);
	cv::Mat gradY = cv::Mat::zeros(11, 11, CV_64F);

	cv::line(gradX, {5, 0}, {5, 10}, 1);
	cv::line(gradY, {0, 5}, {10, 5}, 1);

	const float angle = M_PI / 4;
	patch.setFlowDir(angle);

	common::Pose2d warp = Sophus::SE2d::rot(M_PI / 4);
	patch.setWarp(warp);
	patch.setGrad(gradX, gradY);
	patch.warpImage();

	const auto image = patch.getPredictedNabla();

	for (int i = 1; i < 10; ++i)
	{
		for (int j = 1; j < 10; ++j)
		{
			if (i == j || i == 10 - j)
			{
				EXPECT_LE(image.at<double>(i, j), 0);
			}
		}
	}

#ifdef SAVE_IMAGES
	cv::Mat grayImage;

	double minVal;
	double maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);
	image.convertTo(grayImage, CV_8U, 255.0 / (maxVal - minVal),
					-minVal * 255.0 / (maxVal - minVal));
	cv::imwrite("/tmp/" +
					std::string(::testing::UnitTest::GetInstance()
									->current_test_info()
									->name()) +
					".png",
				grayImage);
#endif
}
