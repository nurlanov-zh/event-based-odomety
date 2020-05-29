#include <dataset_reader/davis240c_reader.h>
#include <replayer/replayer.h>

#include <gtest/gtest.h>
#include <memory>

std::string TEST_DATA_PATH = "test/test_data";

class TestListener
{
   public:
	TestListener() {}
	void eventCallback(const common::Sample<common::Event>& sample)
	{
		timestamps.emplace_back(
			std::make_pair(sample.timestamp, tools::EventType::EVENT));
	}

	void groundTruthCallback(const common::Sample<common::Pose3d>& sample)
	{
		timestamps.emplace_back(
			std::make_pair(sample.timestamp, tools::EventType::GROUND_TRUTH));
	}

	void imageCallback(const common::Sample<cv::Mat>& sample)
	{
		timestamps.emplace_back(
			std::make_pair(sample.timestamp, tools::EventType::IMAGE));
	}

	std::vector<std::pair<common::timestamp_t, tools::EventType>> timestamps;
};

class ReplayerTest : public ::testing::Test
{
   protected:
	void SetUp() override
	{
		std::shared_ptr<tools::DatasetReader> reader =
			std::make_shared<tools::Davis240cReader>(TEST_DATA_PATH);

		replayer.reset(new tools::Replayer(reader));
		replayer->addEventCallback(
			REGISTER_CALLBACK(TestListener, eventCallback, listener));
		replayer->addImageCallback(
			REGISTER_CALLBACK(TestListener, imageCallback, listener));
		replayer->addGroundTruthCallback(
			REGISTER_CALLBACK(TestListener, groundTruthCallback, listener));
	}

	TestListener listener;
    std::unique_ptr<tools::Replayer> replayer;
};

TEST_F(ReplayerTest, nextTest)
{
	while (!replayer->finished()) {
		replayer->next();
	}

	std::vector<std::pair<common::timestamp_t, tools::EventType>> timestamps = {
		{common::timestamp_t(0), tools::EventType::EVENT},
		{common::timestamp_t(1), tools::EventType::IMAGE},
		{common::timestamp_t(3), tools::EventType::EVENT},
		{common::timestamp_t(4), tools::EventType::IMAGE}};

	ASSERT_EQ(timestamps.size(), listener.timestamps.size());
	for (size_t i = 0; i < timestamps.size(); ++i) {
		EXPECT_EQ(timestamps[i].first, listener.timestamps[i].first);
		EXPECT_EQ(timestamps[i].second, listener.timestamps[i].second);
	}
}

TEST_F(ReplayerTest, nextIntervalTest)
{
	replayer->nextInterval(common::timestamp_t(3));

	std::vector<std::pair<common::timestamp_t, tools::EventType>> timestamps = {
		{common::timestamp_t(0), tools::EventType::EVENT},
		{common::timestamp_t(1), tools::EventType::IMAGE},
		{common::timestamp_t(3), tools::EventType::EVENT}};

	ASSERT_EQ(timestamps.size(), listener.timestamps.size());
	for (size_t i = 0; i < timestamps.size(); ++i) {
		EXPECT_EQ(timestamps[i].first, listener.timestamps[i].first);
		EXPECT_EQ(timestamps[i].second, listener.timestamps[i].second);
	}
}