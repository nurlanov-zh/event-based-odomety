#include "visual_odometry/keyframe.h"

namespace visual_odometry
{
Keyframe::Keyframe(const tracker::Patches& patches,
				   const common::timestamp_t& timestamp)
	: timestamp(timestamp)
{
	for (const auto& patch : patches)
	{
		const auto corner = patch.toCorner();
		landmarks_[patch.getTrackId()] = Eigen::Vector2d(corner.x, corner.y);
	}
}

std::vector<tracker::TrackId> Keyframe::getSharedTracks(
	const Keyframe& frame) const
{
	std::vector<tracker::TrackId> tracks;
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