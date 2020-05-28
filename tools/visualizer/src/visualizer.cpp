#include "visualizer/visualizer.h"

#include <replayer/replayer.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

namespace tools
{
constexpr int UI_WIDTH	= 200;
constexpr int IMAGE_VIEWS = 3;

Visualizer::Visualizer() {}

void Visualizer::createWindow()
{
	pangolin::CreateWindowAndBind("Main", 1800, 1000);

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
		// iv->extern_draw_function =
		// 	std::bind(&draw_image_overlay, std::placeholders::_1, idx);
	}
}

void Visualizer::drawImages(const cv::Mat& cvImage)
{
	const pangolin::Image<uint8_t> image((uint8_t*)cvImage.data, cvImage.cols,
										 cvImage.rows, cvImage.cols);
	imgView_[0]->SetImage(image);
	pangolin::FinishFrame();
}
}  // ns tools

int main(int argc, char** argv)
{
	bool showGui		= true;
	std::string dataset = "../data/DAVIS240C/shapes_rotation";

	CLI::App app{"Event based odometry"};

	app.add_option("--show-gui", showGui, "Show GUI");
	app.add_option("--dataset", dataset, "Dataset. Default: " + dataset);

	try
	{
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError& e)
	{
		return app.exit(e);
	}

	tools::Visualizer visualizer;

	if (showGui) {
		visualizer.createWindow();
	}

	tools::Replayer replayer(dataset);

	size_t idx = 0;

	while (!visualizer.shouldQuit() && !replayer.finished()) {
		if (showGui) {
			const auto imageSample = replayer.getCurrentImage();
			visualizer.drawImages(imageSample.value);
		}

		replayer.next();
		replayer.wait();
	}

	return 0;
}
