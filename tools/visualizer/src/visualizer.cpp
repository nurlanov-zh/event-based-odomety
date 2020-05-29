#include "visualizer/visualizer.h"

#include <chrono>
#include <thread>

namespace tools
{
constexpr int VISUALIZER_WIDTH  = 1800;
constexpr int VISUALIZER_HEIGHT = 1000;
constexpr int UI_WIDTH			= 200;
constexpr int IMAGE_VIEWS		= 3;

const std::chrono::milliseconds SLEEP_MILLISECONDS =
	std::chrono::milliseconds(50);

pangolin::Var<bool> stopPlayButton("ui.stopPlay", true, true);
pangolin::Var<bool> nextStepButton("ui.nextStep", false, false);
pangolin::Var<bool> nextIntervalStepButton("ui.nextIntervalStep", false, false);
pangolin::Var<int> stepInterval("ui.stepInterval", 50, 0, 100000);

Visualizer::Visualizer() : quit_(false), currentTimestamp_(0) {}

void Visualizer::createWindow()
{
	pangolin::CreateWindowAndBind("Main", VISUALIZER_WIDTH, VISUALIZER_HEIGHT);

	glEnable(GL_DEPTH_TEST);

	// main parent display for images and 3d viewer
	pangolin::View& mainView =
		pangolin::Display("main")
			.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
			.SetLayout(pangolin::LayoutEqualVertical);

	pangolin::View& imgViewDisplay =
		pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
	mainView.AddDisplay(imgViewDisplay);

	// main ui panel
	pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
										  pangolin::Attach::Pix(UI_WIDTH));

	// 2D image views
	while (imgView_.size() < IMAGE_VIEWS) {
		std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

		size_t idx = imgView_.size();
		imgView_.push_back(iv);

		imgViewDisplay.AddDisplay(*iv);
		iv->extern_draw_function = std::bind(&Visualizer::drawImageOverlay,
											 this, std::placeholders::_1, idx);
	}
}

void Visualizer::drawOriginalImage(const cv::Mat& cvImage)
{
	drawImage(cvImage, ImageViews::ORIGINAL);
}

void Visualizer::drawPredictedFlow(const cv::Mat& cvImage)
{
	drawImage(cvImage, ImageViews::PREDICTED_FLOW);
}

void Visualizer::drawIntegratedFlow(const cv::Mat& cvImage)
{
	drawImage(cvImage, ImageViews::INTEGRATED_FLOW);
}

void Visualizer::drawImage(const cv::Mat& cvImage, const ImageViews& view)
{
	const pangolin::Image<uint8_t> image((uint8_t*)cvImage.data, cvImage.cols,
										 cvImage.rows, cvImage.cols);
	imgView_[static_cast<size_t>(view)]->SetImage(image);
}

void Visualizer::drawImageOverlay(pangolin::View&, size_t idx)
{
	glLineWidth(1.0);
	glColor3f(1.0, 0.0, 0.0);  // red
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	switch (idx)
	{
		case static_cast<size_t>(ImageViews::ORIGINAL):
			drawOriginalOverlay();
			break;

		case static_cast<size_t>(ImageViews::PREDICTED_FLOW):
			drawPredictedFlow();
			break;

		case static_cast<size_t>(ImageViews::INTEGRATED_FLOW):
			drawIntegratedFlow();
			break;
	}
}

void Visualizer::drawOriginalOverlay()
{

	pangolin::GlFont::I().Text("TS: %d", currentTimestamp_.count()).Draw(5, 5);
	pangolin::GlFont::I().Text("Detected %d corners", corners_.size()).Draw(15, 5);

	for (const auto& corner : corners_) {
		pangolin::glDrawCirclePerimeter(corner.point.x(), corner.point.y(), 3.0);

		Eigen::Vector2d r(3, 0);
		Eigen::Rotation2Dd rot(corner.angle);
		r = rot * r;

		pangolin::glDrawLine(corner.point, corner.point + r);
	}
}

void Visualizer::drawPredictedFlow()
{
	pangolin::GlFont::I().Text("Predicted flow").Draw(5, 5);
}

void Visualizer::drawIntegratedFlow()
{
	pangolin::GlFont::I().Text("Integrated flow").Draw(5, 5);
}

void Visualizer::wait() const
{
	std::this_thread::sleep_for(SLEEP_MILLISECONDS);
}

void Visualizer::step()
{
	wait();
	finishVisualizerIteration();
}

bool Visualizer::stopPressed() const
{
	return stopPlayButton;
}

bool Visualizer::nextPressed() const
{
	return pangolin::Pushed(nextStepButton);
}

bool Visualizer::nextIntervalPressed() const
{
	return pangolin::Pushed(nextIntervalStepButton);
}

common::timestamp_t Visualizer::getStepInterval() const
{
	return common::timestamp_t(stepInterval);
}

void Visualizer::finishVisualizerIteration()
{
	pangolin::FinishFrame();
	quit_ = pangolin::ShouldQuit();
}

}  // ns tools
