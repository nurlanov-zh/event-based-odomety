#include "visualizer/visualizer.h"
#include "visualizer/scene_helper.h"

#include <pangolin/display/display_internal.h>

#include <chrono>
#include <thread>

namespace pangolin
{
extern __thread PangolinGl* context;
}

namespace tools
{
constexpr int VISUALIZER_WIDTH = 1800;
constexpr int VISUALIZER_HEIGHT = 1000;
constexpr int UI_WIDTH = 200;
constexpr int IMAGE_VIEWS = 4;

const std::chrono::microseconds SLEEP_MICROSECONDS =
	std::chrono::microseconds(100);

const std::chrono::microseconds INTEGRATION_TIME =
	std::chrono::microseconds(500);

Visualizer::Visualizer()
{
	reset();
}

void Visualizer::createWindow()
{
	pangolin::CreateWindowAndBind("Main", VISUALIZER_WIDTH, VISUALIZER_HEIGHT);

	glEnable(GL_DEPTH_TEST);

	/*********** Main view ***********/
	pangolin::View& mainView =
		pangolin::Display("main")
			.SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
			.SetLayout(pangolin::LayoutEqualVertical);

	pangolin::View& imgViewDisplay =
		pangolin::Display("images").SetLayout(pangolin::LayoutEqualHorizontal);
	mainView.AddDisplay(imgViewDisplay);

	/*********** Create scene ***********/
	const std::string sceneName = "scene";
	sceneView_.reset(new pangolin::View());
	camera_.reset(new pangolin::OpenGlRenderState(
		pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
		pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
								  pangolin::AxisNegY)));

	sceneView_->SetHandler(new pangolin::Handler3D(*(camera_.get())));
	pangolin::context->named_managed_views[sceneName] = sceneView_.get();
	pangolin::context->base.views.push_back(sceneView_.get());

	mainView.AddDisplay(*(sceneView_.get()));

	/*********** UI panel ***********/
	pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
										  pangolin::Attach::Pix(UI_WIDTH));

	/*********** Settings panel ***********/
	const std::string panelName = "settings";
	settingsPanel_.reset(new pangolin::Panel(panelName));
	settingsPanel_->SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
							  pangolin::Attach::Pix(2 * UI_WIDTH));
	pangolin::context->named_managed_views[panelName] = settingsPanel_.get();
	pangolin::context->base.views.push_back(settingsPanel_.get());
	showSettingsPanel_->Meta().gui_changed = true;

	/*********** 2D image views ***********/
	for (int idx = 0; idx < IMAGE_VIEWS; ++idx)
	{
		std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

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
	drawImage(convertImageToGray(cvImage), ImageViews::PREDICTED_NABLA);
}

void Visualizer::drawIntegratedNabla(const cv::Mat& cvImage)
{
	cv::Mat grayImage = convertImageToGray(cvImage);
	cv::Mat imColor;
	applyColorMap(grayImage, imColor, cv::COLORMAP_JET);

	pangolin::GlTexture texture(imColor.cols, imColor.rows, GL_RGB, false, 0,
								GL_RGB, GL_UNSIGNED_BYTE);
	texture.Upload(imColor.data, GL_BGR, GL_UNSIGNED_BYTE);
	imgView_[static_cast<size_t>(ImageViews::INTEGRATED_NABLA)]->SetImage(texture);
	drawImage(convertImageToGray(cvImage), ImageViews::INTEGRATED_NABLA);
}

void Visualizer::drawCostMap(const cv::Mat& cvImage)
{
	cv::Mat grayImage = convertImageToGray(cvImage);
	cv::Mat imColor;
	applyColorMap(grayImage, imColor, cv::COLORMAP_JET);

	pangolin::GlTexture texture(imColor.cols, imColor.rows, GL_RGB, false, 0,
								GL_RGB, GL_UNSIGNED_BYTE);
	texture.Upload(imColor.data, GL_BGR, GL_UNSIGNED_BYTE);
	imgView_[static_cast<size_t>(ImageViews::COST_MAP)]->SetImage(texture);
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
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	switch (idx)
	{
		case static_cast<size_t>(ImageViews::ORIGINAL):
			drawOriginalOverlay();
			break;

		case static_cast<size_t>(ImageViews::PREDICTED_NABLA):
			drawPredictedNablaOverlay();
			break;

		case static_cast<size_t>(ImageViews::INTEGRATED_NABLA):
			drawIntegratedNablaOverlay();
			break;
		case static_cast<size_t>(ImageViews::COST_MAP):
			drawCostMapOverlay();
			break;
	}
}

void Visualizer::drawOriginalOverlay()
{
	const float radius = 2.0;
	pangolin::GlFont::I().Text("TS: %d", currentTimestamp_.count()).Draw(5, 5);
	pangolin::GlFont::I()
		.Text("Detected %d corners", patches_.size())
		.Draw(5, 15);

	for (const auto& patch : patches_)
	{
		if (!patch.isLost())
		{
			const auto& point = patch.toCorner();
			glColor3f(0.0, 1.0, 0.0);  // green
			pangolin::glDrawCirclePerimeter(point.x, point.y, radius);
			pangolin::GlFont::I()
				.Text("%d", patch.getTrackId())
				.Draw(point.x, point.y);
			drawTrajectory(patch);
		}
	}

	// Control mouse click
	if (imgView_[ImageViews::ORIGINAL]->MousePressed())
	{
		const auto selection = imgView_[ImageViews::ORIGINAL]->GetSelection();
		// TODO add logger debug which patch is clicked as soon as patch id come
		// up add this info to images layouts as well
		for (const auto& patch : patches_)
		{
			const auto& point = patch.toCorner();
			if (std::abs(point.x - selection.x.min) <= radius &&
				std::abs(point.y - selection.y.min) <= radius)
			{
				integratedNabla_ = patch.getIntegratedNabla();
				predictedNabla_ = patch.getPredictedNabla();
				costMap_ = patch.getCostMap();
				flow_ = patch.getFlow();
				newPatch_ = patch.getPatch();
				initPatch_ = patch.getInitPatch();
				break;
			}
		}
	}
	else if (nextImagePressed_ || nextPressed_ || nextIntervalPressed_ ||
			 !stopPressed())
	{
		if (patches_.size() > 0)
		{
			integratedNabla_ = patches_.front().getIntegratedNabla();
			predictedNabla_ = patches_.front().getPredictedNabla();
			costMap_ = patches_.front().getCostMap();
			flow_ = patches_.front().getFlow();
			newPatch_ = patches_.front().getPatch();
			initPatch_ = patches_.front().getInitPatch();
		}
	}

	// Draw events
	std::vector<Eigen::Vector2d> positiveEvents;
	std::vector<Eigen::Vector2d> negativeEvents;

	for (const auto& event : integratedEvents_)
	{
		Eigen::Vector2d point(event.value.point.x, event.value.point.y);
		if (event.value.sign == common::EventPolarity::POSITIVE)
		{
			positiveEvents.push_back(point);
		}
		else
		{  // event.value.sign == common::EventPolarity::NEGATIVE
			negativeEvents.push_back(point);
		}
	}

	glColor3f(1.0, 0.0, 0.0);  // red
	pangolin::glDrawPoints(positiveEvents);

	glColor3f(0.0, 0.0, 1.0);  // blue
	pangolin::glDrawPoints(negativeEvents);

	drawIntegratedNabla(integratedNabla_);
	drawPredictedNabla(predictedNabla_);
	drawCostMap(costMap_);
}

void Visualizer::drawScene()
{
	glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
	sceneView_->Activate(*(camera_.get()));
	const u_int8_t colorCameraLeft[3]{0, 125, 0};
	renderCamera(Eigen::Matrix4d::Identity(), 3.f, colorCameraLeft, 0.1f);
}

void Visualizer::drawTrajectory(const tracker::Patch& patch)
{
	const auto& trajectory = patch.getTrajectory();
	glColor3f(0.5, 0.0, 0.5); // purple
	for (int i = static_cast<int>(trajectory.size()) - 1;
		 i > static_cast<int>(trajectory.size()) - 7 && i >= 1; --i)
	{
		pangolin::glDrawLine(
			Eigen::Vector2d(trajectory[i].value.x, trajectory[i].value.y),
			Eigen::Vector2d(trajectory[i - 1].value.x,
							trajectory[i - 1].value.y));
	}
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

cv::Mat Visualizer::convertImageToGray(const cv::Mat& cvImage)
{
	cv::Mat grayImage;

	double minVal;
	double maxVal;
	cv::minMaxLoc(cvImage, &minVal, &maxVal);
	cvImage.convertTo(grayImage, CV_8U, 255.0 / (maxVal - minVal),
					  -minVal * 255.0 / (maxVal - minVal));
	return grayImage;
}

void Visualizer::imageCallback(const common::ImageSample& sample)
{
	originalImage_ = sample.value;
}

void Visualizer::drawPredictedNablaOverlay()
{
	pangolin::GlFont::I().Text("Predicted nabla").Draw(5, 5);
	const auto start =
		Eigen::Vector2d(17 - 5 * std::sin(flow_), 17 - 5 * std::cos(flow_));
	const auto end =
		Eigen::Vector2d(17 + 5 * std::sin(flow_), 17 + 5 * std::cos(flow_));
	const auto arrow1Start =
		Eigen::Vector2d(17 + 5 * std::sin(flow_), 17 + 5 * std::cos(flow_));
	const auto arrow1End = Eigen::Vector2d(17 + 2 * std::sin(flow_ + 0.1),
										   17 + 2 * std::cos(flow_ + 0.1));
	const auto arrow2Start =
		Eigen::Vector2d(17 + 5 * std::sin(flow_), 17 + 5 * std::cos(flow_));
	const auto arrow2End = Eigen::Vector2d(17 + 2 * std::sin(flow_ - 0.1),
										   17 + 2 * std::cos(flow_ - 0.1));
	glColor3f(0.5f, 0.f, 0.2f);
	pangolin::glDrawLine(start, end);
	pangolin::glDrawLine(arrow1Start, arrow1End);
	pangolin::glDrawLine(arrow2Start, arrow2End);

	glColor3f(1.0f, 0.0f, 0.0f);
	pangolin::glDrawCross(Eigen::Vector2d(*patchExtent_, *patchExtent_), 2);
}

void Visualizer::drawIntegratedNablaOverlay()
{
	pangolin::GlFont::I().Text("Integrated nabla").Draw(5, 5);
	glColor3f(1.0f, 0.0f, 0.0f);
	pangolin::glDrawCross(Eigen::Vector2d(*patchExtent_, *patchExtent_), 2);
}

void Visualizer::drawCostMapOverlay()
{
	glColor3f(1.0f, 0.0f, 0.0f);
	pangolin::glDrawCross(Eigen::Vector2d(*patchExtent_, *patchExtent_), 2);

	glColor3f(1.0f, 0.0f, 0.0f);
	const auto x = initPatch_.x - newPatch_.x;
	const auto y = initPatch_.y - newPatch_.y;	
	pangolin::glDrawCross(Eigen::Vector2d(*patchExtent_ - x, *patchExtent_ - y),
						  2);
}

void Visualizer::wait() const
{
	std::this_thread::sleep_for(SLEEP_MICROSECONDS);
}

void Visualizer::step()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// wait();
	if (resetPressed())
	{
		reset();
	}

	drawOriginalImage(originalImage_);
	drawScene();

	finishVisualizerIteration();
}

void Visualizer::reset()
{
	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");

	while (!integratedEvents_.empty())
	{
		integratedEvents_.erase(integratedEvents_.begin());
	}
	setTimestamp(common::timestamp_t(0));
	integratedNabla_ = cv::Mat::zeros(1, 1, CV_64F);
	predictedNabla_ = cv::Mat::zeros(1, 1, CV_64F);
	originalImage_ = cv::Mat::zeros(1, 1, CV_64F);
	costMap_ = cv::Mat::zeros(1, 1, CV_64F);
	quit_ = false;
	nextPressed_ = false;
	nextIntervalPressed_ = false;
	nextImagePressed_ = false;
	flow_ = 0;

	stopPlayButton_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("ui.stopPlay", true, true));
	nextStepButton_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("ui.nextStep", false, false));
	nextIntervalStepButton_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("ui.nextIntervalStep", false, false));
	stepInterval_ = std::unique_ptr<pangolin::Var<int>>(
		new pangolin::Var<int>("ui.stepInterval", 1000, 0, 10000));
	nextImageButton_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("ui.stepImage", false, false));
	resetButton_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("ui.reset", false, false));
	showSettingsPanel_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("ui.showSettings", false, true));

	patchExtent_ = std::unique_ptr<pangolin::Var<int>>(new pangolin::Var<int>(
		"settings.patchExtent", trackerParams_.patchExtent, 0, 50));
	minDistance_ =
		std::unique_ptr<pangolin::Var<double>>(new pangolin::Var<double>(
			"settings.minDistance", trackerParams_.minDistance, 1, 10));
}

bool Visualizer::stopPressed() const
{
	return *stopPlayButton_;
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
	return pangolin::Pushed(*resetButton_);
}

bool Visualizer::nextImagePressed() const
{
	return nextImagePressed_;
}

common::timestamp_t Visualizer::getStepInterval() const
{
	return common::timestamp_t(*stepInterval_);
}

void Visualizer::finishVisualizerIteration()
{
	pangolin::FinishFrame();
	quit_ = pangolin::ShouldQuit();
	nextPressed_ = pangolin::Pushed(*nextStepButton_);
	nextIntervalPressed_ = pangolin::Pushed(*nextIntervalStepButton_);
	nextImagePressed_ = pangolin::Pushed(*nextImageButton_);

	if (showSettingsPanel_->GuiChanged())
	{
		settingsPanel_->Show(*showSettingsPanel_);
	}

	trackerParams_.patchExtent = (*patchExtent_);
	trackerParams_.minDistance = (*minDistance_);
}

}  // namespace tools
