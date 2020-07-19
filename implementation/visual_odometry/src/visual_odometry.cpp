#include "visual_odometry/visual_odometry.h"
#include "visual_odometry/aligner.h"
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

namespace Sophus
{
// apply sim3 to se3 transformation (returns se3)
template <typename Sim3Derived, typename SE3Derived>
Sophus::SE3<typename Eigen::ScalarBinaryOpTraits<
	typename Sim3Derived::Scalar, typename SE3Derived::Scalar>::ReturnType>
operator*(const Sophus::Sim3Base<Sim3Derived>& a,
		  const Sophus::SE3Base<SE3Derived>& b)
{
	return {a.quaternion().normalized() * b.unit_quaternion(),
			a.rxso3() * b.translation() + a.translation()};
}
}  // namespace Sophus

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
	Match match;
	if (!isNewKeyframeNeeded(keyframe, match))
	{
		withoutAdd_++;
		return;
	}

	const auto poseGt = syncGtAndImage(keyframe.timestamp);
	if (poseGt.has_value())
	{
		if (gt_.size() == 0)
		{
			zeroGt_ = poseGt.value();
		}
		gt_.push_back(zeroGt_.inverse() * poseGt.value());
		gtAligned_.push_back(zeroGt_.inverse() * poseGt.value());
	}

	deleteKeyframe();
	addKeyframe(keyframe, match);

	optimize();

	std::list<Keyframe> pose = storedFrames_;
	for (const auto& kf : activeFrames_)
	{
		pose.emplace_back(kf.second);
	}
	if (pose.size() > 5 && gt_.size() > 0)
	{
		gtAligned_.clear();
		visual_odometry::ErrorMetricValue ate;
		const auto sim = align_cameras_sim3(gt_, pose, &ate);
		for (auto& kf : gt_)
		{
			gtAligned_.push_back(sim.inverse() * kf);
		}
		consoleLog_->info("RMSE: " + std::to_string(ate.rmse));
		consoleLog_->info("Mean: " + std::to_string(ate.mean));
		consoleLog_->info("Max: " + std::to_string(ate.max));
		consoleLog_->info("Min: " + std::to_string(ate.min));
	}

	consoleLog_->info("New keyframe is added " +
					  std::to_string(keyframe.timestamp.count()));
	consoleLog_->info("Map consist of " +
					  std::to_string(mapLandmarks_.landmarks.size()) +
					  " landmarks");
}

bool VisualOdometryFrontEnd::isNewKeyframeNeeded(Keyframe& keyframe,
												 Match& match)
{
	if (activeFrames_.empty())
	{
		keyframe.pose = common::Pose3d();
		for (const auto& lm : keyframe.getLandmarks())
		{
			match.inliers.emplace_back(lm.first);
		}

		return true;
	}

	if (activeFrames_.size() == 1)
	{
		if (initCameras(keyframe, match))
		{
			return true;
		}
		return false;
	}

	localizeCamera(keyframe, match);
	keyframe.pose = match.Tw2c;

	if (match.inliers.size() > params_.numOfInliers)
	{
		return true;
	}
	else if (initCameras(keyframe, match))
	{
		return true;
	}
	else if (params_.maxNumWithoutAdd > withoutAdd_)
	{
		match.Tw2c = activeFrames_.rbegin()->second.pose;
		for (const auto& lm : keyframe.getLandmarks())
		{
			match.inliers.emplace_back(lm.first);
		}
		return true;
	}

	consoleLog_->info("Few inliers after localize camera: " +
					  std::to_string(match.inliers.size()));

	return true;
}

void VisualOdometryFrontEnd::addKeyframe(const Keyframe& keyframe,
										 const Match& match)
{
	withoutAdd_ = 0;
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
	const auto startKeyframe = activeFrames_.rbegin()->second;
	getCommonBearingVectors(startKeyframe, keyframe, trackIds, bearingVectors1,
							bearingVectors2);

	const auto inliers = findInliersRansac(bearingVectors1, bearingVectors2,
										   trackIds, keyframe, match);

	consoleLog_->info("Init frames with " + std::to_string(inliers) +
					  " Ransac inliers");
	if (inliers < params_.numOfInliers)
	{
		return false;
	}
	// because Tw2c is relative transform in this case
	keyframe.pose = startKeyframe.pose * match.Tw2c;

	// findInliersEssential(bearingVectors1, bearingVectors2,
	// 					 activeFrames_.begin()->second, keyframe, trackIds,
	// 					 match, 1e-3);

	// consoleLog_->info("Init frames with " +
	// 				  std::to_string(match.inliers.size()) +
	// 				  " Essential inliers");
	// if (match.inliers.size() < params_.numOfEssentialInliers)
	// {
	// 	return false;
	// }

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
								 CentralRelativePoseSacProblem::EIGHTPT));

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
	for (const auto landmark : match.inliers)
	{
		mapLandmarks_.observations[landmark].push_back(
			keyframe.timestamp.count());

		const auto observationIt = mapLandmarks_.observations.find(landmark);

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
			const auto lmIt = mapLandmarks_.landmarks.find(it->first);
			if (lmIt != mapLandmarks_.landmarks.end())
			{
				storedLandmarks_.emplace_back(*lmIt);
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

	size_t i = 0;
	for (auto it = activeFrames_.begin(); it != activeFrames_.end(); ++it, ++i)
	{
		problem.AddParameterBlock(it->second.pose.data(),
								  Sophus::SE3d::num_parameters,
								  new Sophus::test::LocalParameterizationSE3);

		if (i < 2)
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
		if (landmarkIt->second.size() < 2)
		{
			continue;
		}

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
	}

	ceres::Solver::Options ceresOptions;
	ceresOptions.max_num_iterations = params_.maxNumIterations;
	ceresOptions.linear_solver_type = ceres::SPARSE_SCHUR;
	ceresOptions.num_threads = 1;
	ceres::Solver::Summary summary;
	Solve(ceresOptions, &problem, &summary);

	consoleLog_->info("Bundle adjustment optimization info: ");
	consoleLog_->info(summary.BriefReport());
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

std::vector<std::pair<tracker::TrackId, Eigen::Vector3d>> const&
VisualOdometryFrontEnd::getStoredLandmarks() const
{
	return storedLandmarks_;
}

}  // namespace visual_odometry