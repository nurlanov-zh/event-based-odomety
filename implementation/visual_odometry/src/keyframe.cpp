#include "visual_odometry/keyframe.h"

namespace visual_odometry
{
Keyframe::Keyframe() {}

std::vector<size_t> Keyframe::getSharedTracks(Keyframe& frame) const
{
	std::vector<size_t> tracks;
	const auto& frameLandmarks = frame.getLandmarks();
	for (auto it = landmarks_.begin(); it != landmarks_.end(); ++it)
	{
		const auto frameLandmarkIt = frameLandmarks.find(it->first);
		if (frameLandmarkIt == frameLandmarks.end())
		{
			continue;
		}
		tracks.push_back(frameLandmarkIt->first);
	}
	return tracks;
}
}  // ns visual_odometry