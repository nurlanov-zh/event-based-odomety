#pragma once

#include <inttypes.h>
#include <sophus/se3.hpp>

namespace common
{
struct Point2i
{
	int32_t x;
	int32_t y;
};

using Pose3d = Sophus::SE3d;

}  // namespace common