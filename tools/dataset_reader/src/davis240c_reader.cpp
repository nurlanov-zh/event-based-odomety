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

// TODO: you can make it faster...

Davis240cReader::Davis240cReader(const std::string& path) : DatasetReader(path)
{
	consoleLog_ = spdlog::get("console");
	errLog_		= spdlog::get("stderr");
}

EventSequence Davis240cReader::getEvents() const
{
	EventSequence events;
	const std::string filePath = path_ + SEPARATOR + EVENT_FILE;
	std::ifstream file(filePath);

	if (!file.is_open()) {
		errLog_->error("Failed to open {}", filePath);
		throw std::runtime_error("Unable to open " + filePath);
	}

	std::string line;
	size_t loaded = 0;
	while (std::getline(file, line, ' ')) {
		const auto duration = std::chrono::duration<double>(std::stod(line));
		const timestamp_t timestamp =
			std::chrono::duration_cast<timestamp_t>(duration);

		std::getline(file, line, ' ');

		Point2i point;

		point.x = std::stoi(line);

		std::getline(file, line, ' ');

		point.y = std::stoi(line);

		std::getline(file, line);

		int32_t sign = std::stoi(line);

		common::EventPolarity polarity = POSITIVE;
		if (sign == 0) {
			polarity = NEGATIVE;
		}
		else if (sign != 1)
		{
			throw std::runtime_error("Sign is not equal to 0/1");
		}

		const Event event = {point, polarity};
		events.emplace_back(EventSample(event, timestamp));
		loaded++;
	}

	consoleLog_->info("{} event objects are loaded", loaded);

	return events;
}

ImageSequence Davis240cReader::getImages() const
{
	ImageSequence images;
	const std::string filePath = path_ + SEPARATOR + IMAGE_FILE;
	std::ifstream file(filePath);

	if (!file.is_open()) {
		errLog_->error("Failed to open {}", filePath);
		throw std::runtime_error("Unable to open " + filePath);
	}

	std::string line;
	size_t loaded = 0;
	while (std::getline(file, line, ' ')) {
		const auto duration = std::chrono::duration<double>(std::stod(line));
		const timestamp_t timestamp =
			std::chrono::duration_cast<timestamp_t>(duration);

		std::getline(file, line);

		const std::string inputFilePath = path_ + SEPARATOR + line;
		const cv::Mat image				= cv::imread(inputFilePath, CV_8U);
		images.emplace_back(ImageSample(image, timestamp));
		loaded++;
	}

	consoleLog_->info("{} images objects are loaded", loaded);

	return images;
}

GroundTruth Davis240cReader::getGroundTruth() const
{
	GroundTruth groundTruth;
	const std::string filePath = path_ + SEPARATOR + GROUND_TRUTH_FILE;
	std::ifstream file(filePath);

	if (!file.is_open()) {
		errLog_->error("Failed to open {}", filePath);
		throw std::runtime_error("Unable to open " + filePath);
	}

	std::string line;
	size_t loaded = 0;
	while (std::getline(file, line, ' ')) {
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

	consoleLog_->info("{} ground truth objects are loaded", loaded);

	return groundTruth;
}

}  // namespace tools