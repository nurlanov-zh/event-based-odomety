#pragma once

#include "common/geometry.h"

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

	Sample() {}

	T value;
	timestamp_t timestamp;
};

enum EventPolarity
{
	NEGATIVE = -1,
	POSITIVE = 1
};

struct Event
{
	Point2i point;
	EventPolarity sign;
};

using EventSample		= Sample<Event>;
using ImageSample		= Sample<cv::Mat>;
using GroundTruthSample = Sample<Pose3d>;

using EventSequence = std::vector<EventSample>;
using ImageSequence = std::vector<ImageSample>;
using GroundTruth   = std::vector<GroundTruthSample>;

}  // namespace common