#include "visual_odometry/visual_odometry.h"
#include "visual_odometry/local_parameterization_se3.hpp"
#include "visual_odometry/reprojection_error.h"
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

#include <ceres/ceres.h>

#include <cmath>

namespace visual_odometry
{
VisualOdometryFrontEnd::VisualOdometryFrontEnd(
	const common::CameraModelParams<double>& calibration,
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
		Match match;
		for (const auto& lm : keyframe.getLandmarks())
		{
			match.inliers.emplace_back(lm.first);
		}

		addKeyframe(keyframe, match);
		return;
	}

	if (activeFrames_.size() == 1)
	{
		Match match;
		if (initCameras(keyframe, match))
		{
			addKeyframe(keyframe, match);
		}
		consoleLog_->info("Map consist of " +
						  std::to_string(mapLandmarks_.landmarks.size()) +
						  " landmarks");
		return;
	}

	Match match;
	if (!isNewKeyframeNeeded(keyframe, match))
	{
		return;
	}

	deleteKeyframe();
	addKeyframe(keyframe, match);

	// optimize();

	consoleLog_->info("New keyframe is added " +
					  std::to_string(keyframe.timestamp.count()));
	consoleLog_->info("Map consist of " +
					  std::to_string(mapLandmarks_.landmarks.size()) +
					  " landmarks");
}

bool VisualOdometryFrontEnd::isNewKeyframeNeeded(Keyframe& keyframe,
												 Match& match)
{
	localizeCamera(keyframe, match);
	keyframe.pose = match.Tw2c;

	if (match.inliers.size() <= params_.numOfInliers)
	{
		consoleLog_->info("Too few inliers after localize camera " +
						  std::to_string(match.inliers.size()));
		return false;
	}

	return true;
}

void VisualOdometryFrontEnd::addKeyframe(const Keyframe& keyframe,
										 const Match& match)
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

bool VisualOdometryFrontEnd::initCameras(Keyframe& keyframe, Match& match)
{
	opengv::bearingVectors_t bearingVectors1;
	opengv::bearingVectors_t bearingVectors2;
	std::vector<tracker::TrackId> trackIds;
	getCommonBearingVectors(activeFrames_.begin()->second, keyframe, trackIds,
							bearingVectors1, bearingVectors2);

	const auto inliers = findInliersRansac(bearingVectors1, bearingVectors2,
										   trackIds, keyframe, match);

	consoleLog_->info("Init frames with " + std::to_string(inliers) +
					  " Ransac inliers");
	if (inliers < params_.numOfInliers)
	{
		return false;
	}
	keyframe.pose = match.Tw2c;

	findInliersEssential(bearingVectors1, bearingVectors2,
						 activeFrames_.begin()->second, keyframe, trackIds,
						 match, 1e-3);

	consoleLog_->info("Init frames with " + std::to_string(match.inliers.size()) +
					  " Essential inliers");
	if (match.inliers.size() < params_.numOfInliers)
	{
		return false;
	}

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
			consoleLog_->trace("Track is not in landmarks");
			continue;
		}
		points.push_back(landmarkIt->second);

		trackIds.push_back(landmark.first);

		bearingVectors.emplace_back(cameraModel_->unproject(landmark.second));
	}

	opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors,
														  points);

	const float threshold =
		1.0 - std::cos(std::atan2(params_.reprojectionError, 200.));

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
			// std::cout << trackIds[inlier] << std::endl;
			// TODO
			match.inliers.emplace_back(trackIds[inlier]);
		}
	}
	else
	{
		errLog_->error("Ransac is not successful!");
		return;
	}
}

size_t VisualOdometryFrontEnd::findInliersRansac(
	const opengv::bearingVectors_t& bearingVectors1,
	const opengv::bearingVectors_t& bearingVectors2,
	const std::vector<tracker::TrackId>& trackIds, Keyframe& keyframe,
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
		ransac.inliers_.size() >= params_.ransacMinInliers)
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

		match.inliers.reserve(updatedInliers.size());
		for (const auto& inlier : updatedInliers)
		{
			match.inliers.emplace_back(trackIds[inlier]);
		}

		return updatedInliers.size();
	}
	return 0;
}

void VisualOdometryFrontEnd::addNewLandmarks(const Keyframe& keyframe,
											 const Match& match)
{
	std::cout << "Inliers " << match.inliers.size() << std::endl;
	for (const auto landmark : match.inliers)
	{
		mapLandmarks_.observations[landmark].push_back(
			keyframe.timestamp.count());

		const auto observationIt = mapLandmarks_.observations.find(landmark);
		std::cout << "Track " << landmark << " " << observationIt->second.size()
				  << std::endl;
		if (observationIt->second.size() == 2)
		{
			const auto kId1 = observationIt->second.front();
			const auto kId2 = observationIt->second.back();

			const auto pose1 = activeFrames_[kId1].pose;
			const auto pose2 = activeFrames_[kId2].pose;

			const auto pose2d1 =
				activeFrames_[kId1].getLandmarks().at(landmark);
			const auto pose2d2 =
				activeFrames_[kId2].getLandmarks().at(landmark);

			opengv::bearingVectors_t vectors1 = {
				cameraModel_->unproject(pose2d1)};
			opengv::bearingVectors_t vectors2 = {
				cameraModel_->unproject(pose2d2)};

			const auto& positions =
				triangulateLandmarks(pose1, pose2, vectors1, vectors2);

			mapLandmarks_.landmarks[landmark] = positions[0];
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
			const auto obsIt = std::find(it->second.begin(), it->second.end(),
										 keyframe.timestamp.count());
			if (obsIt != it->second.end())
			{
				it->second.erase(obsIt);
			}
		}
	}

	for (auto it = mapLandmarks_.observations.begin();
		 it != mapLandmarks_.observations.end();)
	{
		if (it->second.size() == 0)
		{
			if (mapLandmarks_.landmarks.find(it->first) !=
				mapLandmarks_.landmarks.end())
			{
				mapLandmarks_.landmarks.erase(it->first);
			}
			it = mapLandmarks_.observations.erase(it);
			// TODO save old landmarks
		}
		else
		{
			++it;
		}
	}
}

void VisualOdometryFrontEnd::optimize()
{
	ceres::Problem problem;

	problem.AddParameterBlock(cameraModel_->getParams(), 9);
	problem.SetParameterBlockConstant(cameraModel_->getParams());

	for (auto it = activeFrames_.begin(); it != activeFrames_.end(); ++it)
	{
		problem.AddParameterBlock(it->second.pose.data(),
								  Sophus::SE3d::num_parameters,
								  new Sophus::test::LocalParameterizationSE3);

		if (it == activeFrames_.begin())
		{
			problem.SetParameterBlockConstant(it->second.pose.data());
		}
	}

	for (auto it = mapLandmarks_.landmarks.begin();
		 it != mapLandmarks_.landmarks.end(); ++it)
	{
		problem.AddParameterBlock(it->second.data(), 3);
	}

	for (auto landmarkIt = mapLandmarks_.observations.begin();
		 landmarkIt != mapLandmarks_.observations.end(); ++landmarkIt)
	{
		for (auto observationIt = landmarkIt->second.begin();
			 observationIt != landmarkIt->second.end(); ++observationIt)
		{
			const auto frameIt = activeFrames_.find(*observationIt);
			if (frameIt == activeFrames_.end())
			{
				consoleLog_->trace("Frame is not found");
				continue;
			}

			const auto cornerIt =
				frameIt->second.getLandmarks().find(landmarkIt->first);
			if (cornerIt == frameIt->second.getLandmarks().end())
			{
				consoleLog_->trace("Corner is not found");
				continue;
			}

			const auto pointIt =
				mapLandmarks_.landmarks.find(landmarkIt->first);
			if (pointIt == mapLandmarks_.landmarks.end())
			{
				consoleLog_->trace("Point is not found in the map");
				continue;
			}

			ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
				BundleAdjustmentReprojectionCostFunctor, 2,
				Sophus::SE3d::num_parameters, 3, 9>(
				new BundleAdjustmentReprojectionCostFunctor(cornerIt->second));

			problem.AddResidualBlock(
				costFunction, new ceres::HuberLoss(params_.huberLoss),
				frameIt->second.pose.data(), pointIt->second.data(),
				cameraModel_->getParams());
		}

		ceres::Solver::Options ceresOptions;
		ceresOptions.max_num_iterations = params_.maxNumIterations;
		ceresOptions.linear_solver_type = ceres::SPARSE_SCHUR;
		ceresOptions.num_threads = 1;
		ceres::Solver::Summary summary;
		Solve(ceresOptions, &problem, &summary);

		consoleLog_->debug(summary.BriefReport());
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

std::map<size_t, Keyframe> const& VisualOdometryFrontEnd::getActiveFrames()
	const
{
	return activeFrames_;
}

std::list<Keyframe> const& VisualOdometryFrontEnd::getStoredFrames() const
{
	return storedFrames_;
}

}  // namespace visual_odometry