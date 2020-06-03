#include "dataset_reader/davis240c_reader.h"

#include <chrono>
#include <exception>

namespace tools
{
std::string EVENT_FILE		  = "events.txt";
std::string GROUND_TRUTH_FILE = "groundtruth.txt";
std::string IMAGE_FILE		  = "images.txt";
std::string SEPARATOR		  = "/";

using namespace common;

EventSample Davis240cReader::getEventSample(std::string& line) const
{
	size_t pos = line.find(' ');
	const auto duration =
		std::chrono::duration<double>(std::stod(line.substr(0, pos)));
	const timestamp_t timestamp =
		std::chrono::duration_cast<timestamp_t>(duration);
	line = line.substr(pos + 1);

	Point2i point;
	pos		= line.find(' ');
	point.x = std::stoi(line.substr(0, pos));
	line	= line.substr(pos + 1);

	pos		= line.find(' ');
	point.y = std::stoi(line.substr(0, pos));
	line	= line.substr(pos + 1);

	const int32_t sign = std::stoi(line.substr(0, pos));

	common::EventPolarity polarity = POSITIVE;
	if (sign == 0) { polarity = NEGATIVE; }
	else if (sign != 1)
	{
		throw std::runtime_error("Sign is not equal to 0/1");
	}

	const Event event = {point, polarity};
	return {event, timestamp};
}

ImageSample Davis240cReader::getImageSample(std::string& line) const
{
	size_t pos = line.find(' ');
	const auto duration =
		std::chrono::duration<double>(std::stod(line.substr(0, pos)));

	const timestamp_t timestamp =
		std::chrono::duration_cast<timestamp_t>(duration);
	line = line.substr(pos + 1);

	const std::string inputFilePath = path_ + SEPARATOR + line;
	const cv::Mat image				= cv::imread(inputFilePath, CV_8U);

	return {image, timestamp};
}

GroundTruthSample Davis240cReader::getGroundTruthSample(std::string& line) const
{
	size_t pos = line.find(' ');

	if (!file.is_open())
	{
		errLog_->error("Failed to open {}", filePath);
		throw std::runtime_error("Unable to open " + filePath);
	}

	std::string line;
	size_t loaded = 0;
	while (std::getline(file, line, ' '))
	{
		const auto duration = std::chrono::duration<double>(std::stod(line));
		const timestamp_t timestamp =
			std::chrono::duration_cast<timestamp_t>(duration);

	pos				= line.find(' ');
	const double tx = std::stod(line.substr(0, pos));
	line			= line.substr(pos + 1);

	pos				= line.find(' ');
	const double ty = std::stod(line.substr(0, pos));
	line			= line.substr(pos + 1);

	pos				= line.find(' ');
	const double tz = std::stod(line.substr(0, pos));
	line			= line.substr(pos + 1);

	pos				= line.find(' ');
	const double qx = std::stod(line.substr(0, pos));
	line			= line.substr(pos + 1);

	pos				= line.find(' ');
	const double qy = std::stod(line.substr(0, pos));
	line			= line.substr(pos + 1);

	pos				= line.find(' ');
	const double qz = std::stod(line.substr(0, pos));
	line			= line.substr(pos + 1);

	const double w = std::stod(line);

		common::EventPolarity polarity = POSITIVE;
		if (sign == 0)
		{
			polarity = NEGATIVE;
		}
		else if (sign != 1)
		{
			throw std::runtime_error("Sign is not equal to 0/1");
		}

	const auto pose = Sophus::SE3d(quaternion, translation);
	return {pose, timestamp};
}

Davis240cReader::Davis240cReader(const std::string& path) : DatasetReader(path)
{
	consoleLog_ = spdlog::get("console");
	errLog_		= spdlog::get("stderr");
}

EventSequence Davis240cReader::getEvents() const
{
	const std::string filePath = path_ + SEPARATOR + EVENT_FILE;

	const auto begin  = std::chrono::high_resolution_clock::now();
	const auto events = readFile<EventSequence, EventSample>(
		filePath, std::bind(&Davis240cReader::getEventSample, this,
							std::placeholders::_1));
	const auto end = std::chrono::high_resolution_clock::now();

	consoleLog_->info(
		"{} event objects are loaded in {} milliseconds", events.size(),
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
			.count());

	return events;
}

ImageSequence Davis240cReader::getImages() const
{
	const std::string filePath = path_ + SEPARATOR + IMAGE_FILE;

	if (!file.is_open())
	{
		errLog_->error("Failed to open {}", filePath);
		throw std::runtime_error("Unable to open " + filePath);
	}

	std::string line;
	size_t loaded = 0;
	while (std::getline(file, line, ' '))
	{
		const auto duration = std::chrono::duration<double>(std::stod(line));
		const timestamp_t timestamp =
			std::chrono::duration_cast<timestamp_t>(duration);

	consoleLog_->info(
		"{} images objects are loaded in {} milliseconds", images.size(),
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
			.count());

	return images;
}

GroundTruth Davis240cReader::getGroundTruth() const
{
	const std::string filePath = path_ + SEPARATOR + GROUND_TRUTH_FILE;
	std::ifstream file(filePath);

	if (!file.is_open())
	{
		errLog_->error("Failed to open {}", filePath);
		throw std::runtime_error("Unable to open " + filePath);
	}

	std::string line;
	size_t loaded = 0;
	while (std::getline(file, line, ' '))
	{
		const auto duration = std::chrono::duration<double>(std::stod(line));
		const timestamp_t timestamp =
			std::chrono::duration_cast<timestamp_t>(duration);

		std::getline(file, line, ' ');
		const double tx = std::stod(line);

		std::getline(file, line, ' ');
		const double ty = std::stod(line);

		std::getline(file, line, ' ');
		const double tz = std::stod(line);

		std::getline(file, line, ' ');
		const double qx = std::stod(line);

		std::getline(file, line, ' ');
		const double qy = std::stod(line);

		std::getline(file, line, ' ');
		const double qz = std::stod(line);

		std::getline(file, line);
		const double w = std::stod(line);

		Sophus::SE3d::Point translation(tx, ty, tz);
		Eigen::Quaterniond quaternion(w, qx, qy, qz);

		const auto pose = Sophus::SE3d(quaternion, translation);

		groundTruth.emplace_back(GroundTruthSample(pose, timestamp));
		loaded++;
	}

	const auto begin	   = std::chrono::high_resolution_clock::now();
	const auto groundTruth = readFile<GroundTruth, GroundTruthSample>(
		filePath, std::bind(&Davis240cReader::getGroundTruthSample, this,
							std::placeholders::_1));
	const auto end = std::chrono::high_resolution_clock::now();

	consoleLog_->info(
		"{} ground truth objects are loaded in {} milliseconds",
		groundTruth.size(),
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
			.count());

	return groundTruth;
}

}  // namespace tools