#pragma once

#include "common/geometry.h"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <vector>

namespace common
{
using timestamp_t = std::chrono::microseconds;

template <typename T>
struct Sample
{
	Sample(const T& value, const timestamp_t timestamp)
		: value(value), timestamp(timestamp)
	{
	}

	T value;
	timestamp_t timestamp;
};

struct Event
{
	Point2i point;
	int8_t sign;
};

using EventSequence = std::vector<Sample<Event>>;
using ImageSequence = std::vector<Sample<cv::Mat>>;
using GroundTruth   = std::vector<Sample<Pose3d>>;

}  // namespace common