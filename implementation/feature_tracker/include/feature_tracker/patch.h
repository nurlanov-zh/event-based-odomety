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
	Patch(const Corner& corner, int extent,
		  const common::timestamp_t& timestamp);

	void init();

	void addEvent(const common::EventSample& event);

	void integrateEvents();

	void integrateMotionCompensatedEvents();

	void resetBatch();

	void updateNumOfEvents();

	void addTrajectoryPosition();

	void addFinalCost(double finalCost);

	void updatePatchRect();

	Corner toCorner() const;

	bool isInPatch(const common::Point2i& point) const;

	bool isReady() const;

	bool isLost() const { return lost_; }

	void warpImage();

	common::EventSequence const& getEvents() const;
	cv::Mat const& getIntegratedNabla() const { return integratedNabla_; }
	cv::Mat const& getPredictedNabla() const { return predictedNabla_; }
	cv::Rect2d const& getPatch() const { return patch_; }
	TrackId getTrackId() const { return trackId_; }
	const common::Pose2d& getWarp() const { return warp_; }
	float getFlow() const { return flowDir_; }
	cv::Mat getNormalizedIntegratedNabla() const;
	const cv::Mat& getCostMap() const { return costMap_; }
	bool isInit() { return init_; }
	std::vector<common::Sample<common::Point2d>> const& getTrajectory() const;
	size_t getNumOfEvents() const { return numOfEvents_; }
	cv::Rect2d getInitPatch() const;
	common::timestamp_t getCurrentTimestamp() const;
	common::timestamp_t getTimeWithoutUpdate() const;

	const std::vector<double>& getFinalCosts() const { return finalCosts_; }
	common::timestamp_t getTimeLastUpdate() const;

	cv::Mat const& getCompenatedIntegratedNabla() const;
	common::timestamp_t getInitTime() const { return initTime_; }
	cv::Mat const& getGradX() const;
	cv::Mat const& getGradY() const;

	void setLost() { lost_ = true; }
	void setNumOfEvents(size_t numOfEvents);
	void setTrackId(TrackId trackId) { trackId_ = trackId; }
	void setFlowDir(const double flowDir);
	void setWarp(const common::Pose2d& warp);
	void setCostMap(const cv::Mat& costMap) { costMap_ = costMap; }
	void setIntegratedNabla(const cv::Mat& integratedNabla);
	void setCorner(const Corner& corner, const common::timestamp_t& timestamp);
	void setGrad(const cv::Mat& gradX, const cv::Mat& gradY);
	void setTs(const common::timestamp_t& ts) { currentTimestamp_ = ts; }

	void setMotionCompensatedIntegratedNabla(
		const cv::Mat& motionCompensatedIntegratedNabla);

	void setTimeWithoutUpdate(const common::timestamp_t& timeWithoutUpdate)
	{
		timeWithoutUpdate_ = timeWithoutUpdate;
	}

   private:
	common::Point2i patchToFrameCoords(
		const common::Point2i& pointInPatch) const;

	common::Point2i frameToPatchCoords(
		const common::Point2i& pointInFrame) const;

   private:
	bool init_;
	bool lost_;
	size_t patchId_;
	cv::Rect2d patch_;
	common::Point2d initPoint_;
	TrackId trackId_;

	common::timestamp_t currentTimestamp_;
	common::timestamp_t timeLastUpdate_;
	common::timestamp_t timeWithoutUpdate_;
	common::timestamp_t initTime_;

	common::EventSequence events_;
	size_t numOfEvents_;
	size_t minNumOfEvents_ = 200;
	size_t maxNumOfEvents_ = 350;
	size_t counter_;

	cv::Mat integratedNabla_;
	cv::Mat motionCompensatedIntegratedNabla_;
	cv::Mat predictedNabla_;
	cv::Mat costMap_;

	cv::Mat gradX_;
	cv::Mat gradY_;

	double flowDir_;
	common::Pose2d warp_;
	std::vector<common::Sample<common::Point2d>> trajectory_;

	std::vector<double> finalCosts_;
};

using Patches = std::list<Patch>;

}  // namespace tracker