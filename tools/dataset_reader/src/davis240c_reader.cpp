#include "dataset_reader/davis240c_reader.h"

#include <chrono>
#include <exception>

namespace tools
{
const std::string EVENT_FILE = "events.txt";
const std::string GROUND_TRUTH_FILE = "groundtruth.txt";
const std::string IMAGE_FILE = "images.txt";
const std::string CALIBRATION_FILE = "calib.txt";
const std::string TRAJECTORY_FILE = "trajectory.txt";
const std::string SEPARATOR = "/";
constexpr size_t EVENT_LENGTH = 1000000;

using namespace common;

CameraModelParams<double> Davis240cReader::getCalibrationLine(
	std::string& line) const
{
	CameraModelParams<double> params;
	size_t pos = line.find(' ');
	params.fx = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	params.fy = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	params.cx = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	params.cy = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	params.k1 = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	params.k2 = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	params.p1 = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	params.p2 = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	params.k3 = std::stod(line.substr(0, pos));

	return params;
}

EventSample Davis240cReader::getEventSample(std::string& line) const
{
	size_t pos = line.find(' ');
	const auto duration =
		std::chrono::duration<double>(std::stod(line.substr(0, pos)));
	const timestamp_t timestamp =
		std::chrono::duration_cast<timestamp_t>(duration);
	line = line.substr(pos + 1);

	Point2i point;
	pos = line.find(' ');
	point.x = std::stoi(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	point.y = std::stoi(line.substr(0, pos));
	line = line.substr(pos + 1);

	const int32_t sign = std::stoi(line.substr(0, pos));

	common::EventPolarity polarity = POSITIVE;
	if (sign == 0)
	{
		polarity = NEGATIVE;
	}
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
	const cv::Mat image = cv::imread(inputFilePath, CV_8U);

	return {image, timestamp};
}

GroundTruthSample Davis240cReader::getGroundTruthSample(std::string& line) const
{
	size_t pos = line.find(' ');
	const auto duration =
		std::chrono::duration<double>(std::stod(line.substr(0, pos)));

	const timestamp_t timestamp =
		std::chrono::duration_cast<timestamp_t>(duration);
	line = line.substr(pos + 1);

	pos = line.find(' ');
	const double tx = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	const double ty = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	const double tz = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	const double qx = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	const double qy = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	const double qz = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	const double w = std::stod(line);

	Sophus::SE3d::Point translation(tx, ty, tz);
	Eigen::Quaterniond quaternion(w, qx, qy, qz);

	const auto pose = Sophus::SE3d(quaternion, translation);
	return {pose, timestamp};
}

tracker::Patch Davis240cReader::getTrajectoryLine(std::string& line) const
{
	size_t pos = line.find(' ');
	const int32_t id = std::stoi(line.substr(0, pos));
	line = line.substr(pos + 1);

	pos = line.find(' ');
	const auto duration =
		std::chrono::duration<double>(std::stod(line.substr(0, pos)));

	const timestamp_t timestamp =
		std::chrono::duration_cast<timestamp_t>(duration);
	line = line.substr(pos + 1);

	pos = line.find(' ');
	const double x = std::stod(line.substr(0, pos));
	line = line.substr(pos + 1);

	const double y = std::stod(line);

	tracker::Patch patch({x, y}, 1, timestamp);
	patch.setTrackId(id);
	return patch;
}

Davis240cReader::Davis240cReader(const std::string& path) : DatasetReader(path)
{
	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");

	eventStart_ = 0;
}

std::optional<EventSequence> Davis240cReader::getEvents()
{
	const std::string filePath = path_ + SEPARATOR + EVENT_FILE;

	size_t numOfStrings;
	const auto begin = std::chrono::high_resolution_clock::now();
	const auto events = readFile<EventSequence, EventSample>(
		filePath,
		std::bind(&Davis240cReader::getEventSample, this,
				  std::placeholders::_1),
		eventStart_, eventStart_ + EVENT_LENGTH, numOfStrings);
	const auto end = std::chrono::high_resolution_clock::now();

	consoleLog_->info(
		"{} event objects are loaded in {} milliseconds", events.size(),
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
			.count());

	if (numOfStrings == 0)
	{
		return {};
	}

	eventStart_ += numOfStrings;

	return std::make_optional(events);
}

ImageSequence Davis240cReader::getImages() const
{
	const std::string filePath = path_ + SEPARATOR + IMAGE_FILE;

	size_t numOfStrings;
	const auto begin = std::chrono::high_resolution_clock::now();
	const auto images = readFile<ImageSequence, ImageSample>(
		filePath,
		std::bind(&Davis240cReader::getImageSample, this,
				  std::placeholders::_1),
		0, EVENT_LENGTH, numOfStrings);
	const auto end = std::chrono::high_resolution_clock::now();

	consoleLog_->info(
		"{} images objects are loaded in {} milliseconds", images.size(),
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
			.count());

	return images;
}

GroundTruth Davis240cReader::getGroundTruth() const
{
	const std::string filePath = path_ + SEPARATOR + GROUND_TRUTH_FILE;

	size_t numOfStrings;
	const auto begin = std::chrono::high_resolution_clock::now();
	const auto groundTruth = readFile<GroundTruth, GroundTruthSample>(
		filePath,
		std::bind(&Davis240cReader::getGroundTruthSample, this,
				  std::placeholders::_1),
		0, EVENT_LENGTH, numOfStrings);
	const auto end = std::chrono::high_resolution_clock::now();

	consoleLog_->info(
		"{} ground truth objects are loaded in {} milliseconds",
		groundTruth.size(),
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
			.count());

	return groundTruth;
}

CameraModelParams<double> Davis240cReader::getCalibration() const
{
	const std::string filePath = path_ + SEPARATOR + CALIBRATION_FILE;

	size_t numOfStrings;
	const auto begin = std::chrono::high_resolution_clock::now();
	const auto calibration = readFile<std::vector<CameraModelParams<double>>,
									  CameraModelParams<double>>(
		filePath,
		std::bind(&Davis240cReader::getCalibrationLine, this,
				  std::placeholders::_1),
		0, EVENT_LENGTH, numOfStrings);
	const auto end = std::chrono::high_resolution_clock::now();

	consoleLog_->info(
		"Calibration is loaded in {} milliseconds",
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
			.count());

	return calibration[0];
}

tracker::Patches Davis240cReader::getTrajectory() const
{
	const std::string filePath = path_ + SEPARATOR + TRAJECTORY_FILE;

	size_t numOfStrings;
	const auto begin = std::chrono::high_resolution_clock::now();
	const auto trajectory = readFile<std::list<tracker::Patch>, tracker::Patch>(
		filePath,
		std::bind(&Davis240cReader::getTrajectoryLine, this,
				  std::placeholders::_1),
		0, EVENT_LENGTH, numOfStrings);
	const auto end = std::chrono::high_resolution_clock::now();

	consoleLog_->info(
		"Trajectory is loaded in {} milliseconds",
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
			.count());

	return trajectory;
}
}  // namespace tools