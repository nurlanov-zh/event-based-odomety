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

Davis240cReader::Davis240cReader(const std::string& path) : DatasetReader(path)
{
}

EventSequence Davis240cReader::getEvents() const
{
	EventSequence events;
	const std::string filePath = path_ + SEPARATOR + EVENT_FILE;
	std::ifstream file(filePath);

	if (!file.is_open())
	{ throw std::runtime_error("Unable to open " + filePath); }

	std::string line;
	size_t loaded = 0;
	while (std::getline(file, line, ' '))
	{
		const auto duration = std::chrono::duration<double>(std::stod(line));
		const timestamp_t timestamp =
			std::chrono::duration_cast<timestamp_t>(duration);

		std::getline(file, line, ' ');

		Point2i point;

		point.x = std::stoi(line);

		std::getline(file, line, ' ');

		point.y = std::stoi(line);

		std::getline(file, line);

		const int32_t sign = std::stoi(line);

		if (sign != 1 && sign != 0)
		{ throw std::runtime_error("Sign is not equal to 0/1"); }

		const Event event = {point, static_cast<int8_t>(sign)};
		events.emplace_back(Sample<Event>(event, timestamp));
		loaded++;
	}

	std::cout << std::to_string(loaded) << " events are loaded" << std::endl;

	return events;
}

ImageSequence Davis240cReader::getImages() const
{
	ImageSequence images;
	const std::string filePath = path_ + SEPARATOR + IMAGE_FILE;
	std::ifstream file(filePath);

	if (!file.is_open())
	{ throw std::runtime_error("Unable to open " + filePath); }

	std::string line;
	size_t loaded = 0;
	while (std::getline(file, line, ' '))
	{
		const auto duration = std::chrono::duration<double>(std::stod(line));
		const timestamp_t timestamp =
			std::chrono::duration_cast<timestamp_t>(duration);

		std::getline(file, line);

		const std::string inputFilePath = path_ + SEPARATOR + line;
		const cv::Mat image				= cv::imread(inputFilePath, CV_8U);
		images.emplace_back(Sample<cv::Mat>(image, timestamp));
		loaded++;
	}

	std::cout << std::to_string(loaded) << " images are loaded" << std::endl;

	return images;
}

GroundTruth Davis240cReader::getGroundTruth() const
{
	GroundTruth groundTruth;
	const std::string filePath = path_ + SEPARATOR + GROUND_TRUTH_FILE;
	std::ifstream file(filePath);

	if (!file.is_open())
	{ throw std::runtime_error("Unable to open " + filePath); }

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

		groundTruth.emplace_back(Sample<common::Pose3d>(pose, timestamp));
		loaded++;
	}

	std::cout << std::to_string(loaded) << " ground truth objects are loaded"
			  << std::endl;

	return groundTruth;
}

}  // namespace tools