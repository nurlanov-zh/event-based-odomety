#include "visual_odometry/visual_odometry.h"

namespace visual_odometry
{
VisualOdometryFrontEnd::VisualOdometryFrontEnd(
	const common::CameraModelParams& calibration)
{
	cameraModel_.reset(new common::CameraModel<double>(calibration));
}

void VisualOdometryFrontEnd::newKeyframe(const Keyframe& keyframe)
{
	activeFrames_.push_back(keyframe);

    if (activeFrames_.size() > 2) {
        activeFrames_.pop_front();
    }
}
}  // ns visual_odometry