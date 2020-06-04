#pragma once
#include <common/data_types.h>
#include <feature_tracker/feature_detector.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace tools
{
enum ImageViews
{
	ORIGINAL		= 0,
	PREDICTED_NABLA  = 1,
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

	void setPatches(const tracker::Patches& patches)
	{
		patches_ = patches;
	}

	common::timestamp_t getStepInterval() const;
	void eventCallback(const common::EventSample& sample);
	void imageCallback(const common::ImageSample& sample);

   private:
	void wait() const;

	void finishVisualizerIteration();

	void drawImage(const cv::Mat& cvImage, const ImageViews& view);
	void drawOriginalOverlay();
	void drawPredictedNabla();
	void drawIntegratedNabla();

	void reset();

   private:
	bool quit_;
	bool nextPressed_;
	bool nextIntervalPressed_;
	bool nextImagePressed_;

	common::timestamp_t currentTimestamp_;
	std::vector<std::shared_ptr<pangolin::ImageView>> imgView_;
	tracker::Patches patches_;

	std::list<common::EventSample> integratedEvents_;

	cv::Mat integratedNabla_;
	cv::Mat predictedNabla_;
	cv::Mat originalImage_;

};

}  // ns tools