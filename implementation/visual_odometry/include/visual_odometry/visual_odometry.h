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
	size_t numOfActiveFrames = 20;
	size_t numOfInliers = 55;
	size_t numOfEssentialInliers = 10;
	size_t ransacMinInliers = 15;
	size_t maxNumIterations = 50;
	size_t maxNumWithoutAdd = 4;
	double ransacThreshold = 5e-5;
	double reprojectionError = 3;
	double huberLoss = 0.8;
};

class VisualOdometryFrontEnd
{
   public:
	VisualOdometryFrontEnd(const common::CameraModelParams<double>& calibration,
						   const VisualOdometryParams& params);

	void newKeyframeCandidate(Keyframe& keyframe);

	std::optional<common::Pose3d> syncGtAndImage(
		const common::timestamp_t& timestamp);

	void setGroundTruthSamples(const common::GroundTruth& groundTruthSamples);

	MapLandmarks const& getMapLandmarks();
	std::map<size_t, Keyframe> const& getActiveFrames() const;
	std::list<Keyframe> const& getStoredFrames() const;
	std::vector<common::Pose3d> const& getGtPoses() const { return gtAligned_; }
	std::vector<std::pair<tracker::TrackId, Eigen::Vector3d>> const& 
	getStoredLandmarks() const;

   private:
	void getCommonBearingVectors(const Keyframe& keyframe1,
								 const Keyframe& keyframe2,
								 std::vector<tracker::TrackId>& trackIds,
								 opengv::bearingVectors_t& bearingVectors1,
								 opengv::bearingVectors_t& bearingVectors2);

	bool isNewKeyframeNeeded(Keyframe& keyframe, Match& match);
	bool initCameras(Keyframe& keyframe, Match& match);
	void localizeCamera(const Keyframe& keyframe, Match& match);
	void addKeyframe(const Keyframe& keyframe, const Match& match);
	void addNewLandmarks(const Keyframe& keyframe, const Match& match);
	void deleteKeyframe();
	void deleteLandmarks(const Keyframe& keyframe);
	size_t findInliersRansac(const opengv::bearingVectors_t& bearingVectors1,
							 const opengv::bearingVectors_t& bearingVectors2,
							 const std::vector<tracker::TrackId>& trackIds,
							 Keyframe& keyframe, Match& match);
	void optimize();

   private:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	VisualOdometryParams params_;

	std::vector<common::Pose3d> gt_;
	std::vector<common::Pose3d> gtAligned_;
	common::Pose3d zeroGt_;

	bool init_;

	std::map<size_t, Keyframe> activeFrames_;
	std::list<Keyframe> storedFrames_;
	MapLandmarks mapLandmarks_;
	std::vector<std::pair<tracker::TrackId, Eigen::Vector3d>> storedLandmarks_;
	std::unique_ptr<common::CameraModel<double>> cameraModel_;
	common::GroundTruth groundTruthSamples_;

	size_t withoutAdd_;
};
}  // namespace visual_odometry