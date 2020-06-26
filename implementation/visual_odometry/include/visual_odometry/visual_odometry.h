#pragma once

#include "visual_odometry/keyframe.h"

#include <common/camera_model.h>
#include <common/data_types.h>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>

namespace std
{
template <typename _rep, typename ratio>
struct hash<std::chrono::duration<_rep, ratio>>
{
	typedef std::chrono::duration<_rep, ratio> argument_type;
	typedef std::size_t result_type;
	result_type operator()(argument_type const& s) const
	{
		return std::hash<_rep>{}(
			static_cast<std::chrono::duration<_rep, std::micro>>(s).count());
	}
};
}  // namespace std

namespace visual_odometry
{
struct VisualOdometryParams
{
	size_t numOfActiveFrames = 10;
	size_t numOfInliers = 50;
	size_t ransacMinInliers = 50;
	double ransacThreshold = 0.1;
	double reprojectionError = 0.1;
};

class VisualOdometryFrontEnd
{
   public:
	VisualOdometryFrontEnd(const common::CameraModelParams& calibration,
						   const VisualOdometryParams& params);

	void newKeyframeCandidate(Keyframe& keyframe);

	std::optional<common::Pose3d> syncGtAndImage(
		const common::timestamp_t& timestamp);

	void setGroundTruthSamples(const common::GroundTruth& groundTruthSamples);

	MapLandmarks const& getMapLandmarks();
	std::list<Keyframe> const& getActiveFrames() const;
	std::list<Keyframe> const& getStoredFrames() const;

   private:
	void getCommonBearingVectors(const Keyframe& keyframe1,
								 const Keyframe& keyframe2,
								 std::vector<tracker::TrackId>& trackIds,
								 opengv::bearingVectors_t& bearingVectors1,
								 opengv::bearingVectors_t& bearingVectors2);

	bool initCameras(Keyframe& keyframe);
	void localizeCamera(const Keyframe& keyframe, Match& match);
	void addKeyframe(const Keyframe& keyframe);
	void deleteKeyframe();
	size_t findInliersRansac(const opengv::bearingVectors_t& bearingVectors1,
							 const opengv::bearingVectors_t& bearingVectors2,
							 Keyframe& keyframe, Match& match);

   private:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	VisualOdometryParams params_;

	bool init_;

	std::list<Keyframe> activeFrames_;
	std::list<Keyframe> storedFrames_;
	MapLandmarks mapLandmarks_;
	std::unique_ptr<common::CameraModel<double>> cameraModel_;
	common::GroundTruth groundTruthSamples_;
};
}  // namespace visual_odometry