#pragma once
#include <common/data_types.h>

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
	PREDICTED_FLOW  = 1,
	INTEGRATED_FLOW = 2
};

class Visualizer
{
   public:
	Visualizer();

	bool shouldQuit() const { return quit_; }

	void createWindow();

	void drawOriginalImage(const cv::Mat& cvImage);
	void drawPredictedFlow(const cv::Mat& cvImage);
	void drawIntegratedFlow(const cv::Mat& cvImage);

	void drawImageOverlay(pangolin::View&, size_t idx);

	void step();

	bool stopPressed() const;
	bool nextPressed() const;
	bool nextIntervalPressed() const;

	void setTimestamp(const common::timestamp_t& timestamp)
	{
		currentTimestamp_ = timestamp;
	}

	void setCorners(const common::Corners& corners)
	{
		corners_ = corners;
	}

	common::timestamp_t getStepInterval() const;

   private:
	void wait() const;

	void finishVisualizerIteration();

	void drawImage(const cv::Mat& cvImage, const ImageViews& view);
	void drawOriginalOverlay();
	void drawPredictedFlow();
	void drawIntegratedFlow();

   private:
	bool quit_;
	common::timestamp_t currentTimestamp_;
	std::vector<std::shared_ptr<pangolin::ImageView>> imgView_;
	common::Corners corners_;
};

}  // ns tools