#include <dataset_reader/davis240c_reader.h>
#include <feature_tracker/feature_detector.h>

#include <gtest/gtest.h>

std::string TEST_DATA_PATH = "test/test_data";

TEST(FeatureDetector, featureDetectorTest)
{
	cv::Mat image = cv::imread(TEST_DATA_PATH + "/test_image.png");

	tracker::DetectorParams params;
	params.patchExtent = 5;
	tracker::FeatureDetector detector(params, image.size());

	const auto corners = detector.detectFeatures(image);
	EXPECT_GT(corners.size(), 10);

	for (const auto& corner : corners) {
		EXPECT_TRUE(corner.x > params.patchExtent + 1 &&
					corner.x < image.cols - params.patchExtent - 1);
		EXPECT_TRUE(corner.y > params.patchExtent + 1 &&
					corner.y < image.rows - params.patchExtent - 1);
	}

#ifdef SAVE_IMAGES
	for (const auto& corner : corners) {
		image.at<cv::Vec3b>(corner.y, corner.x) = {0, 0, 255};
	}

	cv::imwrite("/tmp/" +
					std::string(::testing::UnitTest::GetInstance()
									->current_test_info()
									->name()) +
					".png",
				image);

#endif
}

TEST(FeatureDetector, updatePatchTest)
{
	tracker::DetectorParams params;
	params.patchExtent = 5;
	tracker::FeatureDetector detector(params, {240, 180});

	tracker::Patches patches = {tracker::Patch({0, 0, 11, 11}),
								tracker::Patch({5, 5, 11, 11}),
								tracker::Patch({20, 20, 11, 11})};

	detector.setPatches(patches);

	common::EventSequence events;

	for (size_t i = 0; i < 5; ++i) {
		common::EventSample event;
		event.timestamp   = common::timestamp_t(i);
		event.value.point = {std::rand() % 30, std::rand() % 30};
		event.value.sign  = std::rand() % 2 == 1
							   ? common::EventPolarity::POSITIVE
							   : common::EventPolarity::NEGATIVE;
		detector.updatePatches(event);

		for (size_t j = 0; j < patches.size(); ++j) {
			if (patches[j].isInPatch(event.value.point)) {
				patches[j].addEvent(event);
			}
		}
	}

	const auto detectorPatches = detector.getPatches();

	ASSERT_EQ(detectorPatches.size(), patches.size());

	for (size_t i = 0; i < patches.size(); ++i) {
		const auto detectorEvents = detectorPatches[i].getEvents();
		const auto gtEvents		  = patches[i].getEvents();
		ASSERT_EQ(detectorEvents.size(), gtEvents.size());

		auto detectorIt = detectorEvents.begin();
		for (auto gtIt = gtEvents.begin();
			 gtIt != gtEvents.end() && detectorIt != detectorEvents.end();
			 ++gtIt, ++detectorIt)
		{
			EXPECT_EQ(detectorIt->timestamp, gtIt->timestamp);
		}
	}
}