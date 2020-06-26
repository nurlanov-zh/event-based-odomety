#include "visual_odometry/visual_odometry.h"
#include "visual_odometry/triangulation.h"
#include "visual_odometry/reprojection_error.h"
#include "visual_odometry/local_parameterization_se3.hpp"

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
		activeFrames_[keyframe.timestamp] = keyframe;
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
	keyframe.pose = match.Tw2c;

	if (match.inliers.size() <= params_.numOfInliers)
	{
		return;
	}

	// find inliers
	// localizeCamera(keyframe, match);

	deleteKeyframe();
	addKeyframe(keyframe, match);
	
	optimize();

	consoleLog_->info("New keyframe is added " + std::to_string(keyframe.timestamp.count()));
	consoleLog_->info("Map consist of " + std::to_string(mapLandmarks_.landmarks.size()) +
					  " landmarks");
}

void VisualOdometryFrontEnd::addKeyframe(const Keyframe& keyframe, const Match& match)
{
	activeFrames_[keyframe.timestamp.count()] = keyframe;

	addNewLandmarks(keyframe, match);
}

void VisualOdometryFrontEnd::deleteKeyframe()
{
	if (activeFrames_.size() > params_.numOfActiveFrames)
	{
		// TODO add more sophisticated strategy
		storedFrames_.push_back(activeFrames_.begin()->second);
		deleteLandmarks(activeFrames_.begin()->second);
		activeFrames_.erase(activeFrames_.begin());
	}
}

bool VisualOdometryFrontEnd::initCameras(Keyframe& keyframe)
{
	opengv::bearingVectors_t bearingVectors1;
	opengv::bearingVectors_t bearingVectors2;
	std::vector<tracker::TrackId> trackIds;
	getCommonBearingVectors(activeFrames_.begin()->second, keyframe, trackIds,
							bearingVectors1, bearingVectors2);

	Match match;
	findInliersRansac(bearingVectors1, bearingVectors2, keyframe, match);

	if (match.inliers.size() < params_.numOfInliers)
	{
		return false;
	}

	keyframe.pose = match.Tw2c;
	return true;
}

void VisualOdometryFrontEnd::localizeCamera(const Keyframe& keyframe,
											Match& match)
{
	match.inliers.clear();

	opengv::bearingVectors_t bearingVectors;
	opengv::points_t points;
	std::vector<tracker::TrackId> trackIds;

	for (const auto& landmark : keyframe.getLandmarks())
	{
		const auto landmarkIt = mapLandmarks_.landmarks.find(landmark.first);

		if (landmarkIt == mapLandmarks_.landmarks.end())
		{
			errLog_->error("Track is not in landmarks");
			return;
		}
		points.push_back(landmarkIt->second);

		trackIds.push_back(Landmarks.first);

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
		match.Tw2c = Sophus::SE3d(transformation);

		std::vector<int> updatedInliers;
		ransac.sac_model_->selectWithinDistance(nonlinearTransformation,
												threshold, updatedInliers);

		match.inliers.reserve(updatedInliers.size()); 
		for (const auto& inlier : updatedInliers)
		{
			// TODO
			match.inliers.emplace_back(std::make_pair(trackIds[inlier], 0));
		}
	}
	else
	{
		errorLog_->error("Ransac is not successful!");
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
		match.Tw2c = Sophus::SE3d(transformation);
		match.Tw2c.translation().normalize();

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

void VisualOdometryFrontEnd::addNewLandmarks(const Keyframe& keyframe, const Match& match)
{
	for (const auto& landmark : match.inliers)
	{
		const auto it = mapLandmarks_.landmarks.find(landmark.first);
		if (it == mapLandmarks_.landmarks.end())
		{
			mapLandmarks_.observations[landmark.first].push_back(keyframe.timestamp.count());
			const auto observationIt = mapLandmarks_.observations.find(landmark.first);
			if (observationIt != mapLandmarks_.observations.end())
			{
				if (observationIt->second.size() == 2)
				{
					const auto kId1 = observationIt->second.front();
					const auto kId2 = observationIt->second.back();

					const auto pose1 = activeFrames_[kId1].pose;
					const auto pose2 = activeFrames_[kId2].pose;

					const auto pose2d1 = activeFrames_[kId1].getLandmarks()[landmark.first];
					const auto pose2d2 = activeFrames_[kId2].getLandmarks()[landmark.first];

					opengv::bearingVectors_t vectors1 = {cameraModel_->unproject(pose2d1)};
					opengv::bearingVectors_t vectors2 = {cameraModel_->unproject(pose2d2)};

					const auto& positions = triangulateLandmarks(pose1, pose2, vectors1, vectors2);

					mapLandmarks_.landmarks[landmark.first] = positions[0];
				}
			}
		}
	}
}

void VisualOdometryFrontEnd::deleteLandmarks(const Keyframe& keyframe)
{
	for (const auto& landmarks : keyframe.getLandmarks())
	{
		const auto it = mapLandmarks_.observations.find(landmarks.first);
		if (it != mapLandmarks_.observations.end())
		{
			it->second.erase(keyframe.timestamp.count());
		}
	}

	for (auto it = mapLandmarks_.observations.begin(); it != mapLandmarks_.observations.end(); ++it)
	{
		if (it->second.size() == 0)
		{
			mapLandmarks_.landmarks.erase(it->first);
			mapLandmarks_.observations.erase(it->first);
			// TODO save old landmarks
		}
	}
}

void VisualOdometryFrontEnd::optimize()
{
	ceres::Problem problem;

  	for (auto it = activeFrames_.begin(); it != activeFrames_.end(); ++it) {
    	problem.AddParameterBlock(activeFrames_->second.pose.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
    }
  }

  for (auto it = mapLandmarks_.landmarks.begin(); it != mapLandmarks_.landmarks.end(); ++it) {
    problem.AddParameterBlock(it->second.data(), 3);
  }

  for (auto landmarkIt = mapLandmarks_.observations.begin(); landmarkIt != mapLandmarks_.observations.end();
       ++landmarkIt) {
    for (auto observationIt = landmarkIt->second.begin();
         observationIt != landmarkIt->second.end(); ++observationIt) {
		const auto frameIt = activeFrames_.find(*observationIt);
		if (frameIt == activeFrames_.end())
		{
			errorLog_->error("Frame is not found");
			return;
		}

		const auto cornerIt = frameIt->getLandmarks().find(landmarkIt->first);
		if (cornerIt == frameIt->getLandmarks().end())
		{
			errorLog_->error("Corner is not found");
			return;
		}

		ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
			BundleAdjustmentReprojectionCostFunctor, 2,
			Sophus::SE3d::num_parameters, 3, 8>(
				new BundleAdjustmentReprojectionCostFunctor(
					*cornerIt));

		const auto pointIt = mapLandmarks_.landmarks.find(landmarkIt->first);
		if (pointIt == mapLandmarks_.landmarks.end())
		{
			errorLog_->error("Point is not found in the map");
			return;
		}

		problem.AddResidualBlock(
			costFunction, new ceres::HuberLoss(params_.huberLoss),
			frameIt->pose.data(), pointIt->data(), cameraModel_->getParams());
    }

	ceres::Solver::Options ceres_options;
	ceres_options.max_num_iterations = params_.maxNumIterations;
	ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
	ceres_options.num_threads = 1;
	ceres::Solver::Summary summary;
	Solve(ceres_options, &problem, &summary);

	consoleLog_->debug(summary.BriefReport());
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

std::unordered_map<common::timestamp_t, Keyframe> const& VisualOdometryFrontEnd::getActiveFrames() const
{
	return activeFrames_;
}

std::list<Keyframe> const& VisualOdometryFrontEnd::getStoredFrames() const
{
	return storedFrames_;
}

}  // namespace visual_odometry