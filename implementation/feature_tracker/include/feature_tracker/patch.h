#pragma once

#include <common/data_types.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

namespace tracker
{
using Corner = cv::Point2d;
using Corners = std::vector<Corner>;

class Patch
{
   public:
	Patch();

	Patch(const cv::Rect2i& rect);

	Patch(const Corner& corner, int extent);

	Patch(const Corner& corner, int extent, const size_t num_patches);

	void init();

	void addEvent(const common::EventSample& event);

	void integrateEvents();

	void resetBatch();

	void updateNumOfEvents();

	void optimizePatchParams();

	void updatePatchRect(const common::Pose2d& warp);

	Corner toCorner() const;

	bool isInPatch(const common::Point2i& point) const;

	bool isReady() const { return events_.size() >= numOfEvents_; }

	void warpImage();

	void addTrajectoryPosition(const common::Point2d& pose,
							   common::timestamp_t timestamp)
	{
		trajectory_.push_back({pose, timestamp});
	}

	common::EventSequence const& getEvents() const;

	cv::Mat const& getIntegratedNabla() const { return integratedNabla_; }

	cv::Mat const& getPredictedNabla() const { return predictedNabla_; }

	cv::Rect2i const& getPatch() const { return patch_; }

	size_t getTrackId() const { return trackId_; }

	size_t getPatchId() const { return patchId_; }

	std::vector<common::Sample<common::Point2d>> const& getTrajectory() const
	{
		return trajectory_;
	}

	cv::Mat getNormalizedIntegratedNabla() const;

	const common::Pose2d& getWarp() const { return warp_; }

	float getFlow() const { return flowDir_; }

	void setNumOfEvents(size_t numOfEvents) { numOfEvents_ = numOfEvents; }

	void setGrad(const cv::Mat& gradX, const cv::Mat& gradY)
	{
		gradX_ = gradX;
		gradY_ = gradY;
	}

	void setFlowDir(const double flowDir) { flowDir_ = flowDir; }

	void setWarp(const common::Pose2d& warp) { warp_ = warp; }

	void setTrackId(const size_t trackId) { trackId_ = trackId; }

	void setLost() { lost_ = true; }

	void setIntegratedNabla(const cv::Mat& integratedNabla)
	{
		integratedNabla_ = integratedNabla;
	}

   private:
	common::Point2i patchToFrameCoords(
		const common::Point2i& pointInPatch) const;

	common::Point2i frameToPatchCoords(
		const common::Point2i& pointInFrame) const;

   private:
	size_t patchId_;
	size_t trackId_;
	cv::Rect2i patch_;

	common::EventSequence events_;
	size_t numOfEvents_;
	size_t minNumOfEvents_ = 20;

	bool lost_;

	cv::Mat gradX_;
	cv::Mat gradY_;
	cv::Mat integratedNabla_;
	cv::Mat predictedNabla_;

	double flowDir_;
	common::Pose2d warp_;
	std::vector<common::Sample<common::Point2d>> trajectory_;
};

using Patches = std::vector<Patch>;

}  // namespace tracker