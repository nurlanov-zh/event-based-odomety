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
	INTEGRATED_NABLA = 2
};

class Visualizer
{
   public:
	Visualizer();

	bool shouldQuit() const { return quit_; }

	void createWindow();

	void drawOriginalImage(const cv::Mat& cvImage);
	void drawPredictedNabla(const cv::Mat& cvImage);
	void drawIntegratedNabla(const cv::Mat& cvImage);

	void drawImageOverlay(pangolin::View&, size_t idx);

	void step();

	bool stopPressed() const;
	bool nextPressed() const;
	bool nextIntervalPressed() const;
	bool resetPressed() const;
	bool nextImagePressed() const;

	void setTimestamp(const common::timestamp_t& timestamp)
	{
		currentTimestamp_ = timestamp;
	}

	void setPatches(const tracker::Patches& patches) { patches_ = patches; }

	common::timestamp_t getStepInterval() const;
	void eventCallback(const common::EventSample& sample);
	void imageCallback(const common::ImageSample& sample);

	tracker::DetectorParams const& getTrackerParams() const
	{
		return trackerParams_;
	}

   private:
	void wait() const;

	void finishVisualizerIteration();

	void drawImage(const cv::Mat& cvImage, const ImageViews& view);
	void drawOriginalOverlay();
	void drawPredictedNabla();
	void drawIntegratedNabla();
	void drawTrajectory(const tracker::Patch& patch);

	void reset();

   private:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	bool quit_;
	bool nextPressed_;
	bool nextIntervalPressed_;
	bool nextImagePressed_;

	std::unique_ptr<pangolin::Panel> settingsPanel_;
	std::unique_ptr<pangolin::Var<bool>> stopPlayButton_;
	std::unique_ptr<pangolin::Var<bool>> nextStepButton_;
	std::unique_ptr<pangolin::Var<bool>> nextIntervalStepButton_;
	std::unique_ptr<pangolin::Var<int>> stepInterval_;
	std::unique_ptr<pangolin::Var<bool>> nextImageButton_;
	std::unique_ptr<pangolin::Var<bool>> resetButton_;
	std::unique_ptr<pangolin::Var<bool>> showSettingsPanel_;
	std::unique_ptr<pangolin::Var<int>> patchExtent_;
	std::unique_ptr<pangolin::Var<double>> minDistance_;

	common::timestamp_t currentTimestamp_;
	std::vector<std::shared_ptr<pangolin::ImageView>> imgView_;
	tracker::Patches patches_;

	std::list<common::EventSample> integratedEvents_;

	cv::Mat integratedNabla_;
	cv::Mat predictedNabla_;
	cv::Mat originalImage_;

	tracker::DetectorParams trackerParams_;
};

}  // namespace tools