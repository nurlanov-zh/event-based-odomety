#pragma once

#include <common/data_types.h>

namespace tracker
{
using Corner  = cv::Point2d;
using Corners = std::vector<Corner>;

class Patch
{
   public:
	Patch();

	Patch(const cv::Rect2i& rect);

	Patch(const Corner& corner, int extent);

	void init();

	void addEvent(const common::EventSample& event);

	void integrateEvents();

	void resetBatch();

	Corner toCorner() const;

	bool isInPatch(const common::Point2i& point) const;

	bool isReady() const { return events_.size() >= numOfEvents_; }

	void warpImage();

	common::EventSequence const& getEvents() const;

	cv::Mat const& getIntegratedNabla() const { return integratedNabla_; }

	cv::Mat const& getPredictedNabla() const { return predictedNabla_; }

	cv::Rect2i const& getPatch() const { return patch_; }

	std::vector<common::Sample<common::Pose2d>> const& getTrajectory() const
	{
		return trajectory_;
	}

	void setNumOfEvents(size_t numOfEvents) { numOfEvents_ = numOfEvents; }

	void setGrad(const cv::Mat& gradX, const cv::Mat& gradY)
	{
		gradX_ = gradX;
		gradY_ = gradY;
	}

	void setFlowDir(const float flowDir) { flowDir_ = flowDir; }

	void setWarp(const common::Pose2d& warp) { warp_ = warp; }

	void setLost() { lost_ = true; }

   private:
	common::Point2i patchToFrameCoords(
		const common::Point2i& pointInPatch) const;

	common::Point2i frameToPatchCoords(
		const common::Point2i& pointInFrame) const;

   private:
	cv::Rect2i patch_;

	common::EventSequence events_;
	size_t numOfEvents_;

	bool lost_;

	cv::Mat gradX_;
	cv::Mat gradY_;
	cv::Mat integratedNabla_;
	cv::Mat predictedNabla_;

	float flowDir_;
	common::Pose2d warp_;
	std::vector<common::Sample<common::Pose2d>> trajectory_;
};

using Patches = std::vector<Patch>;

}  // ns tracker