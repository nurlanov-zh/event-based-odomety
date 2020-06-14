#pragma once

#include <common/data_types.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

namespace tracker
{
using Corner = cv::Point2d;
using Corners = std::vector<Corner>;
using TrackId = int32_t;

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

	void updatePatchRect();

	Corner toCorner() const;

	bool isInPatch(const common::Point2i& point) const;

	bool isReady() const { return events_.size() >= numOfEvents_; }

	bool isLost() const { return lost_; }

	void warpImage(const cv::Mat& gradX, const cv::Mat& gradY);

	void addTrajectoryPosition(const common::Point2d& pose,
							   common::timestamp_t timestamp)
	{
		trajectory_.push_back({pose, timestamp});
	}

	common::EventSequence const& getEvents() const;
	cv::Mat const& getIntegratedNabla() const { return integratedNabla_; }
	cv::Mat const& getPredictedNabla() const { return predictedNabla_; }
	cv::Rect2i const& getPatch() const { return patch_; }
	TrackId getTrackId() const { return trackId_; }
	const common::Pose2d& getWarp() const { return warp_; }
	float getFlow() const { return flowDir_; }
	cv::Mat getNormalizedIntegratedNabla() const;
	const cv::Mat& getCostMap() const { return costMap_; }
	const cv::Mat& getCostMap2() const { return costMap2_; }
	bool isInit() { return init_; }
	std::vector<common::Sample<common::Point2d>> const& getTrajectory() const;
	size_t getNumOfEvents() const { return numOfEvents_; }
	cv::Rect2i getInitPatch() const;

	void setLost() { lost_ = true; }
	void setNumOfEvents(size_t numOfEvents);
	void setTrackId(TrackId trackId) { trackId_ = trackId; }
	void setFlowDir(const double flowDir);
	void setWarp(const common::Pose2d& warp);
	void setCostMap(const cv::Mat& costMap) { costMap_ = costMap; }
	void setCostMap2(const cv::Mat& costMap) { costMap2_ = costMap; }
	void setIntegratedNabla(const cv::Mat& integratedNabla);
	void setCorner(const Corner& corner);

   private:
	common::Point2i patchToFrameCoords(
		const common::Point2i& pointInPatch) const;

	common::Point2i frameToPatchCoords(
		const common::Point2i& pointInFrame) const;

   private:
	bool init_;
	bool lost_;
	size_t patchId_;
	cv::Rect2i patch_;
	common::Point2d initPoint_;
	TrackId trackId_;

	common::EventSequence events_;
	size_t numOfEvents_;
	size_t minNumOfEvents_ = 30;
	size_t maxNumOfEvents_ = 500;

	cv::Mat integratedNabla_;
	cv::Mat predictedNabla_;
	cv::Mat costMap_;
	cv::Mat costMap2_;

	double flowDir_;
	common::Pose2d warp_;
	std::vector<common::Sample<common::Point2d>> trajectory_;
};

using Patches = std::list<Patch>;

}  // namespace tracker