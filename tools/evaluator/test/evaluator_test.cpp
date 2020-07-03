#include <evaluator/evaluator.h>

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

TEST(EvaluatorTest, saveTrajectoryTest)
{
	std::list<tracker::Patch> patches;
	std::vector<std::pair<size_t, common::Sample<common::Point2d>>>
		trajectories;

	for (size_t i = 0; i < 2; ++i)
	{
		tracker::Patch patch({0, 0}, 10, common::timestamp_t(0));
		patch.setTrackId(i);
		trajectories.push_back(std::make_pair(
			patch.getTrackId(),
			common::Sample<common::Point2d>({0, 0}, common::timestamp_t(0))));

		for (size_t j = 0; j < 30; ++j)
		{
			common::Sample<common::Point2d> sample;
			sample.value = cv::Point2d(j, j);
			sample.timestamp = common::timestamp_t(j);
			patch.setCorner(sample.value, sample.timestamp);
			trajectories.push_back(std::make_pair(patch.getTrackId(), sample));
		}
		patches.push_back(patch);
	}

	const auto param = tools::EvaluatorParams();
	tools::Evaluator evaluator(param);

	evaluator.saveFeaturesTrajectory(patches);

	std::ifstream file;
	file.open(param.outputDir + "/trajectory.txt");
	EXPECT_TRUE(file);

	std::vector<std::tuple<size_t, double, double, double>> parsedTraj;
	double ts;
	double x;
	double y;
	size_t id;

	while (file >> id >> ts >> x >> y)
	{
		parsedTraj.push_back(std::make_tuple(id, ts, x, y));
	}

	ASSERT_EQ(parsedTraj.size(), trajectories.size());

	for (size_t i = 0; i < parsedTraj.size(); ++i)
	{
		double ts;
		double x;
		double y;
		size_t id;

		std::tie(id, ts, x, y) = parsedTraj[i];

		EXPECT_FLOAT_EQ(
			std::chrono::duration<double>(trajectories[i].second.timestamp)
				.count(),
			ts);
		EXPECT_FLOAT_EQ(x, trajectories[i].second.value.x);
		EXPECT_FLOAT_EQ(y, trajectories[i].second.value.y);
		EXPECT_EQ(id, trajectories[i].first);
	}
}
