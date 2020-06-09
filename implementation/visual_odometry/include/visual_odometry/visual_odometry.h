#pragma once

#include "visual_odometry/keyframe.h"

#include <common/data_types.h>
#include <common/camera_model.h>

namespace std
{
    template<typename _rep, typename ratio>
    struct hash<std::chrono::duration<_rep, ratio>>
    {
        typedef std::chrono::duration<_rep, ratio> argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& s) const
        {
            return std::hash<_rep>{}( static_cast<std::chrono::duration<_rep, std::micro>>(s).count());
        }
    };
}

namespace visual_odometry
{
class VisualOdometryFrontEnd
{
   public:
	VisualOdometryFrontEnd(const common::CameraModelParams& calibration);

    void newKeyframe(const Keyframe& keyframe);

   private:
	std::list<Keyframe> activeFrames_;
	std::list<Keyframe> archivedFrames_;
    std::unique_ptr<common::CameraModel<double>> cameraModel_;
};
}  // namespace visual_odometry