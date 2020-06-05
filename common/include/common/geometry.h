#pragma once

#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

namespace common
{
using Point2i = cv::Point2i;
using Pose3d = Sophus::SE3d;
using Pose2d = Sophus::SE2d;

}  // namespace common