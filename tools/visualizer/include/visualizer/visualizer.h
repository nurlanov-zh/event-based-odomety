#pragma once
#include <common/data_types.h>
#include <feature_tracker/feature_detector.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

namespace tools
{
enum ImageViews
{
	ORIGINAL = 0,
	PREDICTED_NABLA = 1,
	INTEGRATED_NABLA = 2,
	COST_MAP = 3
};

class Visualizer
{
   public:
	Visualizer();

	bool shouldQuit() const { return quit_; }

	void createWindow();

	void step();

	bool stopPressed() const;
	bool nextPressed() const;
	bool nextIntervalPressed() const;
	bool nextImagePressed() const;

	void setTimestamp(const common::timestamp_t& timestamp)
	{
		currentTimestamp_ = timestamp;
	}

	void setPatches(const tracker::Patches& patches) { patches_ = patches; }

	common::timestamp_t getStepInterval() const;
	void eventCallback(const common::EventSample& sample);
	void imageCallback(const common::ImageSample& sample);

	tracker::DetectorParams const& getTrackerParams()
	{
		trackerParamsChanged_ = false;
		return trackerParams_;
	}

	bool isTrackerParamsChanged() const { return trackerParamsChanged_; }

   private:
	void wait() const;

	void finishVisualizerIteration();

	void drawImage(const cv::Mat& cvImage, const ImageViews& view);

	void drawOriginalImage(const cv::Mat& cvImage);
	void drawPredictedNabla(const cv::Mat& cvImage);
	void drawIntegratedNabla(const cv::Mat& cvImage);
	void drawCostMap(const cv::Mat& cvImage);

	void drawImageOverlay(pangolin::View&, size_t idx);
	void drawOriginalOverlay();
	void drawPredictedNablaOverlay();
	void drawIntegratedNablaOverlay();
	void drawCostMapOverlay();
	void drawTrajectory(const tracker::Patch& patch);

	cv::Mat convertImageToGray(const cv::Mat& cvImage);

	void drawScene();

	void reset();

	void updateTrackerParams();

   private:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	bool quit_;
	bool nextPressed_;
	bool nextIntervalPressed_;
	bool nextImagePressed_;
	bool trackerParamsChanged_;

	std::unique_ptr<pangolin::Panel> settingsPanel_;
	std::unique_ptr<pangolin::View> sceneView_;
	std::unique_ptr<pangolin::OpenGlRenderState> camera_;

	std::unique_ptr<pangolin::Var<bool>> stopPlayButton_;
	std::unique_ptr<pangolin::Var<bool>> nextStepButton_;
	std::unique_ptr<pangolin::Var<bool>> nextIntervalStepButton_;
	std::unique_ptr<pangolin::Var<int>> stepInterval_;
	std::unique_ptr<pangolin::Var<bool>> nextImageButton_;
	std::unique_ptr<pangolin::Var<bool>> showSettingsPanel_;
	std::unique_ptr<pangolin::Var<int>> patchExtent_;
	std::unique_ptr<pangolin::Var<double>> minDistance_;
	std::unique_ptr<pangolin::Var<bool>> drawCostMap_;
	std::unique_ptr<pangolin::Var<double>> optimizerThreshold_;
	std::unique_ptr<pangolin::Var<double>> huberLoss_;

	common::timestamp_t currentTimestamp_;
	std::vector<std::shared_ptr<pangolin::ImageView>> imgView_;
	tracker::Patches patches_;

	std::list<common::EventSample> integratedEvents_;

	cv::Mat integratedNabla_;
	cv::Mat costMap_;
	cv::Mat predictedNabla_;
	cv::Mat originalImage_;
	cv::Rect2d newPatch_;
	cv::Rect2d initPatch_;

	tracker::DetectorParams trackerParams_;

	float flow_;
};

}  // namespace tools