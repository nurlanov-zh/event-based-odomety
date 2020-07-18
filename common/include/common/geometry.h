#pragma once

#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>

namespace common
{
using Point2i = cv::Point2i;
using Point2d = cv::Point2d;
using Point3d = cv::Point3d;
using Pose3d = Sophus::SE3d;
using Pose2d = Sophus::SE2d;

}  // namespace common