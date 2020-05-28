#pragma once

#include <pangolin/display/image_view.h>
#include <opencv2/opencv.hpp>

namespace tools
{
class Visualizer
{
   public:
	Visualizer();

	bool shouldQuit() const { return pangolin::ShouldQuit(); }

	void createWindow();

	void drawImages(const cv::Mat& cvImage);

   private:
	std::vector<std::shared_ptr<pangolin::ImageView>> imgView_;
};

}  // ns tools