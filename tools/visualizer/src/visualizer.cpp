#include "visualizer/visualizer.h"

#include <chrono>
#include <thread>

namespace tools
{
constexpr int VISUALIZER_WIDTH  = 1800;
constexpr int VISUALIZER_HEIGHT = 1000;
constexpr int UI_WIDTH			= 200;
constexpr int IMAGE_VIEWS		= 3;

const std::chrono::microseconds SLEEP_MICROSECONDS =
	std::chrono::microseconds(100);

const std::chrono::microseconds INTEGRATION_TIME =
	std::chrono::microseconds(500);

pangolin::Var<bool> stopPlayButton("ui.stopPlay", true, true);
pangolin::Var<bool> nextStepButton("ui.nextStep", false, false);
pangolin::Var<bool> nextIntervalStepButton("ui.nextIntervalStep", false, false);
pangolin::Var<int> stepInterval("ui.stepInterval", 1000, 0, 10000);
pangolin::Var<bool> nextImageButton("ui.stepImage", false, false);
pangolin::Var<bool> resetButton("ui.reset", false, false);

Visualizer::Visualizer()
{
	reset();
}

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

void Visualizer::drawPredictedNabla(const cv::Mat& cvImage)
{
	cv::Mat grayImage;

	double minVal;
	double maxVal;
	cv::minMaxLoc(cvImage, &minVal, &maxVal);
	cvImage.convertTo(grayImage, CV_8U, 255.0 / (maxVal - minVal),
					  -minVal * 255.0 / (maxVal - minVal));

	drawImage(grayImage, ImageViews::PREDICTED_NABLA);
}

void Visualizer::drawIntegratedNabla(const cv::Mat& cvImage)
{
	cv::Mat grayImage;

	double minVal;
	double maxVal;
	cv::minMaxLoc(cvImage, &minVal, &maxVal);
	cvImage.convertTo(grayImage, CV_8U, 255.0 / (maxVal - minVal),
					  -minVal * 255.0 / (maxVal - minVal));

	drawImage(grayImage, ImageViews::INTEGRATED_NABLA);
}

void Visualizer::drawImage(const cv::Mat& cvImage, const ImageViews& view)
{
	const pangolin::Image<uint8_t> image((uint8_t*)cvImage.data, cvImage.cols,
										 cvImage.rows, cvImage.cols);
	imgView_[static_cast<size_t>(view)]->SetImage(image);
}

void Visualizer::drawImageOverlay(pangolin::View& /*view*/, size_t idx)
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

		case static_cast<size_t>(ImageViews::PREDICTED_NABLA):
			drawPredictedNabla();
			break;

		case static_cast<size_t>(ImageViews::INTEGRATED_NABLA):
			drawIntegratedNabla();
			break;
	}
}

void Visualizer::drawOriginalOverlay()
{
	const float radius = 3.0;
	pangolin::GlFont::I().Text("TS: %d", currentTimestamp_.count()).Draw(5, 5);
	pangolin::GlFont::I()
		.Text("Detected %d corners", patches_.size())
		.Draw(5, 15);

	for (const auto& patch : patches_) {
		const auto& point = patch.toCorner();
		pangolin::glDrawCirclePerimeter(point.x, point.y, radius);
	}

	// Control mouse click
	if (imgView_[ImageViews::ORIGINAL]->MousePressed()) {
		const auto selection = imgView_[ImageViews::ORIGINAL]->GetSelection();
		// TODO add logger debug which patch is clicked as soon as patch id come up
		// add this info to images layouts as well
		for (const auto& patch : patches_) {
			const auto& point = patch.toCorner();
			if (std::abs(point.x - selection.x.min) <= radius &&
				std::abs(point.y - selection.y.min) <= radius)
			{
				integratedNabla_ = patch.getIntegratedNabla();
				predictedNabla_  = patch.getPredictedNabla();
				break;
			}
		}
	}
	else if (nextImagePressed_ || nextPressed_ || nextIntervalPressed_ ||
			 !stopPressed())
	{
		if (patches_.size() > 0) {
			integratedNabla_ = patches_.front().getIntegratedNabla();
			predictedNabla_  = patches_.front().getPredictedNabla();
		}
	}

	// Draw events
	std::vector<Eigen::Vector2d> positiveEvents;
	std::vector<Eigen::Vector2d> negativeEvents;

	for (const auto& event : integratedEvents_) {
		Eigen::Vector2d point(event.value.point.x, event.value.point.y);
		if (event.value.sign == common::EventPolarity::POSITIVE) {
			positiveEvents.push_back(point);
		}
		else
		{  // event.value.sign == common::EventPolarity::NEGATIVE
			negativeEvents.push_back(point);
		}
	}

	glColor3f(1.0, 0.0, 0.0);  // red
	pangolin::glDrawPoints(positiveEvents);

	glColor3f(0.0, 1.0, 0.0);  // green
	pangolin::glDrawPoints(negativeEvents);

	drawIntegratedNabla(integratedNabla_);
	drawPredictedNabla(predictedNabla_);
}

void Visualizer::eventCallback(const common::EventSample& sample)
{
	integratedEvents_.emplace_back(sample);

	while (integratedEvents_.back().timestamp -
			   integratedEvents_.front().timestamp >=
		   INTEGRATION_TIME)
	{
		integratedEvents_.erase(integratedEvents_.begin());
	}
}

void Visualizer::imageCallback(const common::ImageSample& sample)
{
	originalImage_ = sample.value;
}

void Visualizer::drawPredictedNabla()
{
	pangolin::GlFont::I().Text("Predicted nabla").Draw(5, 5);
}

void Visualizer::drawIntegratedNabla()
{
	pangolin::GlFont::I().Text("Integrated nabla").Draw(5, 5);
}

void Visualizer::wait() const
{
	std::this_thread::sleep_for(SLEEP_MICROSECONDS);
}

void Visualizer::step()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// wait();
	if (resetPressed()) {
		reset();
	}

	drawOriginalImage(originalImage_);

	finishVisualizerIteration();
}

void Visualizer::reset()
{
	consoleLog_ = spdlog::get("console");
	errLog_		= spdlog::get("stderr");

	while (!integratedEvents_.empty()) {
		integratedEvents_.erase(integratedEvents_.begin());
	}
	setTimestamp(common::timestamp_t(0));
	integratedNabla_	 = cv::Mat::zeros(1, 1, CV_64F);
	predictedNabla_		 = cv::Mat::zeros(1, 1, CV_64F);
	originalImage_		 = cv::Mat::zeros(1, 1, CV_64F);
	quit_				 = false;
	nextPressed_		 = false;
	nextIntervalPressed_ = false;
	nextImagePressed_	= false;
}

bool Visualizer::stopPressed() const
{
	return stopPlayButton;
}

bool Visualizer::nextPressed() const
{
	return nextPressed_;
}

bool Visualizer::nextIntervalPressed() const
{
	return nextIntervalPressed_;
}

bool Visualizer::resetPressed() const
{
	return pangolin::Pushed(resetButton);
}

bool Visualizer::nextImagePressed() const
{
	return nextImagePressed_;
}

common::timestamp_t Visualizer::getStepInterval() const
{
	return common::timestamp_t(stepInterval);
}

void Visualizer::finishVisualizerIteration()
{
	pangolin::FinishFrame();
	quit_				 = pangolin::ShouldQuit();
	nextPressed_		 = pangolin::Pushed(nextStepButton);
	nextIntervalPressed_ = pangolin::Pushed(nextIntervalStepButton);
	nextImagePressed_	= pangolin::Pushed(nextImageButton);
}

}  // ns tools
