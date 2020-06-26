#include "visual_odometry/visual_odometry.h"
#include "visual_odometry/triangulation.h"

#include <sophus/interpolate.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include <cmath>

namespace visual_odometry
{
VisualOdometryFrontEnd::VisualOdometryFrontEnd(
	const common::CameraModelParams& calibration,
	const VisualOdometryParams& params)
	: params_(params)
{
	cameraModel_.reset(new common::CameraModel<double>(calibration));

	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");

	init_ = false;
}

void VisualOdometryFrontEnd::newKeyframeCandidate(Keyframe& keyframe)
{
	if (activeFrames_.empty())
	{
		keyframe.pose = common::Pose3d();
		activeFrames_.push_back(keyframe);
		return;
	}

	if (activeFrames_.size() == 1)
	{
		if (initCameras(keyframe))
		{
			addKeyframe(keyframe);
		}
		return;
	}

	Match match;
	localizeCamera(keyframe, match);

	if (match.inliers.size() > params_.numOfInliers)
	{
		deleteKeyframe();
		addKeyframe(keyframe);
	}

	// optimize

	// for (auto& frame : activeFrames_)
	// {
	// 	const auto firstImagePose = syncGtAndImage(frame.timestamp);
	// 	const auto secondImagePose = syncGtAndImage(keyframe.timestamp);

	// 	if (!firstImagePose.has_value() || !secondImagePose.has_value())
	// 	{
	// 		continue;
	// 	}

	// 	frame.pose = firstImagePose.value();

	// 	if ((firstImagePose.value().translation() -
	// 		 secondImagePose.value().translation())
	// 			.norm() < 0.1)
	// 	{
	// 		continue;
	// 	}

	// 	opengv::bearingVectors_t bearingVectors1;
	// 	opengv::bearingVectors_t bearingVectors2;
	// 	std::vector<tracker::TrackId> trackIds;

	// 	getCommonBearingVectors(frame, keyframe, trackIds, bearingVectors1,
	// 							bearingVectors2);

	// 	const auto landmarks = triangulateLandmarks(
	// 		firstImagePose.value(), secondImagePose.value(), bearingVectors1,
	// 		bearingVectors2);

	// 	for (size_t i = 0; i < trackIds.size(); ++i)
	// 	{
	// 		if (!std::isnan(landmarks[i].x()) &&
	// 			!std::isnan(landmarks[i].y()) && !std::isnan(landmarks[i].z()))
	// 		{
	// 			mapLandmarks_[trackIds[i]] = landmarks[i];
	// 		}
	// 	}
	// }

	consoleLog_->info("Map consist of " + std::to_string(mapLandmarks_.size()) +
					  " landmarks");
}

void VisualOdometryFrontEnd::addKeyframe(const Keyframe& keyframe)
{
	activeFrames_.push_back(keyframe);

	//addNewLandmarks(keyframe);
	// add landmarks
}

void VisualOdometryFrontEnd::deleteKeyframe()
{
	if (activeFrames_.size() > params_.numOfActiveFrames)
	{
		storedFrames_.push_back(activeFrames_.front());
		activeFrames_.pop_front();
		// delete landmarks
	}
}

bool VisualOdometryFrontEnd::initCameras(Keyframe& keyframe)
{
	opengv::bearingVectors_t bearingVectors1;
	opengv::bearingVectors_t bearingVectors2;
	std::vector<tracker::TrackId> trackIds;
	getCommonBearingVectors(activeFrames_.front(), keyframe, trackIds,
							bearingVectors1, bearingVectors2);

	Match match;
	findInliersRansac(bearingVectors1, bearingVectors2, keyframe, match);

	if (match.inliers.size() < params_.numOfInliers)
	{
		return false;
	}

	keyframe.pose = match.T_i_j;
	return true;
}

void VisualOdometryFrontEnd::localizeCamera(const Keyframe& keyframe,
											Match& match)
{
	match.inliers.clear();

	opengv::bearingVectors_t bearingVectors;
	opengv::points_t points;

	for (const auto& landmark : keyframe.getLandmarks())
	{
		const auto landmarkIt = mapLandmarks_.find(landmark.first);

		if (landmarkIt == mapLandmarks_.end())
		{
			errLog_->error("Track is not in landmarks");
			return;
		}
		points.push_back(landmarkIt->second);

		bearingVectors.emplace_back(cameraModel_->unproject(landmark.second));
	}

	opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors,
														  points);

	const float threshold =
		1.0 - std::cos(std::atan2(params_.reprojectionError, 500.));

	opengv::sac::Ransac<
		opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
		ransac;

	std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
		absposeproblem_ptr(
			new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
				adapter, opengv::sac_problems::absolute_pose::
							 AbsolutePoseSacProblem::KNEIP));

	ransac.sac_model_ = absposeproblem_ptr;
	ransac.threshold_ = threshold;

	if (ransac.computeModel())
	{
		adapter.sett(ransac.model_coefficients_.col(3));
		adapter.setR(ransac.model_coefficients_.block<3, 3>(0, 0));

		opengv::transformation_t nonlinearTransformation =
			opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

		Eigen::Matrix<double, 4, 4> transformation =
			Eigen::Matrix<double, 4, 4>::Identity();
		transformation.block<3, 4>(0, 0) = nonlinearTransformation;
		match.T_i_j = Sophus::SE3d(transformation);

		std::vector<int> updatedInliers;
		ransac.sac_model_->selectWithinDistance(nonlinearTransformation,
												threshold, updatedInliers);

		// match.inliers.reserve(updatedInliers.size());
		// for (const auto& inlier : updatedInliers)
		// {
		// 	md.inliers.emplace_back(md.matches[inlier]);
		// }
	}
	else
	{
		// It is better raise exception here, but it is not handeled
		std::cerr << "Ransac is not successful!" << std::endl;
		return;
	}
}

size_t VisualOdometryFrontEnd::findInliersRansac(
	const opengv::bearingVectors_t& bearingVectors1,
	const opengv::bearingVectors_t& bearingVectors2, Keyframe& keyframe,
	Match& match)
{
	match.inliers.clear();

	opengv::relative_pose::CentralRelativeAdapter adapter(bearingVectors1,
														  bearingVectors2);

	opengv::sac::Ransac<
		opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
		ransac;
	std::shared_ptr<
		opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
		relposeproblemPtr(
			new opengv::sac_problems::relative_pose::
				CentralRelativePoseSacProblem(
					adapter, opengv::sac_problems::relative_pose::
								 CentralRelativePoseSacProblem::NISTER));

	ransac.sac_model_ = relposeproblemPtr;
	ransac.threshold_ = params_.ransacThreshold;

	if (ransac.computeModel() &&
		static_cast<int>(ransac.inliers_.size()) >= params_.ransacMinInliers)
	{
		adapter.sett12(ransac.model_coefficients_.col(3));
		adapter.setR12(ransac.model_coefficients_.block<3, 3>(0, 0));

		opengv::transformation_t nonlinearTransformation =
			opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

		Eigen::Matrix<double, 4, 4> transformation =
			Eigen::Matrix<double, 4, 4>::Identity();
		transformation.block<3, 4>(0, 0) = nonlinearTransformation;
		match.T_i_j = Sophus::SE3d(transformation);
		match.T_i_j.translation().normalize();

		std::vector<int> updatedInliers;
		ransac.sac_model_->selectWithinDistance(
			nonlinearTransformation, params_.ransacThreshold, updatedInliers);
		std::vector<int> inliers(updatedInliers.size());

		return updatedInliers.size();
		// for (size_t i = 0; i < updatedInliers.size(); ++i)
		// {
		// 	inliers[i] = md.matches[updatedInliers[i]];
		// }

		// match.inliers = inliers;
	}
}

void VisualOdometryFrontEnd::getCommonBearingVectors(
	const Keyframe& keyframe1, const Keyframe& keyframe2,
	std::vector<tracker::TrackId>& trackIds,
	opengv::bearingVectors_t& bearingVectors1,
	opengv::bearingVectors_t& bearingVectors2)
{
	const auto& sharedTracks = keyframe1.getSharedTracks(keyframe2);
	const auto& landmarks1 = keyframe1.getLandmarks();
	const auto& landmarks2 = keyframe2.getLandmarks();

	for (const auto& track : sharedTracks)
	{
		const auto landmarks1It = landmarks1.find(track);
		const auto landmarks2It = landmarks2.find(track);
		trackIds.push_back(track);

		bearingVectors1.push_back(
			cameraModel_->unproject(landmarks1It->second));
		bearingVectors2.push_back(
			cameraModel_->unproject(landmarks2It->second));
	}
}

std::optional<common::Pose3d> VisualOdometryFrontEnd::syncGtAndImage(
	const common::timestamp_t& timestamp)
{
	auto lowerBoundIt =
		std::lower_bound(groundTruthSamples_.begin(), groundTruthSamples_.end(),
						 common::GroundTruthSample({}, timestamp),
						 [](const common::GroundTruthSample& a,
							const common::GroundTruthSample& b) {
							 return a.timestamp < b.timestamp;
						 });
	if (lowerBoundIt == groundTruthSamples_.end())
	{
		return {};
	}
	const auto nextPose = *lowerBoundIt;

	if (lowerBoundIt->timestamp == timestamp)
	{
		return std::make_optional(lowerBoundIt->value);
	}

	if (lowerBoundIt == groundTruthSamples_.begin())
	{
		return {};
	}

	--lowerBoundIt;
	if (lowerBoundIt == groundTruthSamples_.end())
	{
		return {};
	}
	const auto prevPose = *lowerBoundIt;

	const auto interploatedPose = Sophus::interpolate(
		prevPose.value, nextPose.value,
		static_cast<float>((timestamp - prevPose.timestamp).count()) /
			(nextPose.timestamp - prevPose.timestamp).count());

	return std::make_optional(interploatedPose);
}

void VisualOdometryFrontEnd::setGroundTruthSamples(
	const common::GroundTruth& groundTruthSamples)
{
	groundTruthSamples_ = groundTruthSamples;
}

MapLandmarks const& VisualOdometryFrontEnd::getMapLandmarks()
{
	return mapLandmarks_;
}

std::list<Keyframe> const& VisualOdometryFrontEnd::getActiveFrames() const
{
	return activeFrames_;
}

std::list<Keyframe> const& VisualOdometryFrontEnd::getStoredFrames() const
{
	return storedFrames_;
}

}  // namespace visual_odometry