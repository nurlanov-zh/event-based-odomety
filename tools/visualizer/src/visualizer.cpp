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
		pangolin::ModelViewLookAt(0, 0, 0, 2.1, 0.6, 0.2, pangolin::AxisNegY)));

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

	// drawImage(convertImageToGray(cvImage), ImageViews::COST_MAP);
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
			if (!patch.isLost())
			{
				const auto& point = patch.toCorner();
				if (std::abs(point.x - selection.x.min) <= radius &&
					std::abs(point.y - selection.y.min) <= radius)
				{
					integratedNabla_ = patch.getIntegratedNabla();
					predictedNabla_ = patch.getPredictedNabla();

					costMap_ = patch.getCostMap();

					// costMap_ = patch.getCompenatedIntegratedNabla();

					flow_ = patch.getFlow();
					newPatch_ = patch.getPatch();
					initPatch_ = patch.getInitPatch();
					track_id_ = patch.getTrackId();
					break;
				}
			}
		}
	}
	else if (nextImagePressed_ || nextPressed_ || nextIntervalPressed_ ||
			 !stopPressed())
	{
		for (const auto& patch : patches_)
		{
			if (!patch.isLost())
			{
				integratedNabla_ = patch.getIntegratedNabla();
				predictedNabla_ = patch.getPredictedNabla();

				costMap_ = patch.getCostMap();

				// costMap_ = patch.getCompenatedIntegratedNabla();

				flow_ = patch.getFlow();
				newPatch_ = patch.getPatch();
				initPatch_ = patch.getInitPatch();
				track_id_ = patch.getTrackId();
				break;
			}
		}
/*		integratedNabla_ = integratedEventImage_;
		predictedNabla_ = compensatedEventImage_;*/
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

	const u_int8_t colorCameraActive[3]{255, 0, 0};
	const u_int8_t colorCameraStored[3]{0, 255, 0};
	const u_int8_t colorCameraGt[3]{0, 0, 255};
	const u_int8_t colorActivePoints[3]{0, 0, 0};
	const u_int8_t colorOldPoints[3]{49, 51, 53};

	for (const auto& camera : activeFrames_)
	{
		renderCamera(camera.second.pose.matrix(), 3.f, colorCameraActive, 0.1f);
	}

	for (const auto& camera : gt_)
	{
		renderCamera(camera.matrix(), 3.f, colorCameraGt, 0.1f);
	}

	for (const auto& camera : storedFrames_)
	{
		renderCamera(camera.pose.matrix(), 3.f, colorCameraStored, 0.1f);
	}

	glPointSize(3.0);
	glBegin(GL_POINTS);

	for (const auto& trackLandmark : landmarks_.landmarks)
	{
		glColor3ubv(colorActivePoints);
		pangolin::glVertex(trackLandmark.second);
	}

	glEnd();

	for (const auto& trackLandmark : landmarks_.landmarks)
	{
		pangolin::GlFont::I()
			.Text("%d", trackLandmark.first)
			.Draw(trackLandmark.second(0), trackLandmark.second(1),
				  trackLandmark.second(2));
	}

	glPointSize(3.0);
	glBegin(GL_POINTS);

	for (const auto& lm : storedLandmarks_)
	{
		glColor3ubv(colorOldPoints);
		pangolin::glVertex(lm.second);
	}

	glEnd();

	for (const auto& lm : storedLandmarks_)
	{
		pangolin::GlFont::I()
			.Text("%d", lm.first)
			.Draw(lm.second(0), lm.second(1), lm.second(2));
	}
}

void Visualizer::drawTrajectory(const tracker::Patch& patch)
{
	const auto& trajectory = patch.getTrajectory();
	glColor3f(0.5, 0.0, 0.5);  // purple
	for (int i = static_cast<int>(trajectory.size()) - 1;
		 i > static_cast<int>(trajectory.size()) - 15 && i >= 1; --i)
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
	cvImage.convertTo(grayImage, CV_8U, 255.0 / (maxVal / 2 - minVal),
					  -minVal * 255.0 / (maxVal / 2 - minVal));
	return grayImage;
}

void Visualizer::imageCallback(const common::ImageSample& sample)
{
	set_ = true;
	originalImage_ = sample.value;
}

void Visualizer::drawPredictedNablaOverlay()
{
	pangolin::GlFont::I().Text("Predicted nabla").Draw(1, 1);

	pangolin::GlFont::I().Text("track_id: %d", track_id_).Draw(1, 3);

	const auto start =
		Eigen::Vector2d(17 - 5 * std::cos(flow_), 17 - 5 * std::sin(flow_));
	const auto end =
		Eigen::Vector2d(17 + 5 * std::cos(flow_), 17 + 5 * std::sin(flow_));
	const auto arrow1Start =
		Eigen::Vector2d(17 + 5 * std::cos(flow_), 17 + 5 * std::sin(flow_));
	const auto arrow1End = Eigen::Vector2d(17 + 2 * std::cos(flow_ + 0.1),
										   17 + 2 * std::sin(flow_ + 0.1));
	const auto arrow2Start =
		Eigen::Vector2d(17 + 5 * std::cos(flow_), 17 + 5 * std::sin(flow_));
	const auto arrow2End = Eigen::Vector2d(17 + 2 * std::cos(flow_ - 0.1),
										   17 + 2 * std::sin(flow_ - 0.1));
	glColor3f(0.5f, 0.f, 0.2f);
	pangolin::glDrawLine(start, end);
	pangolin::glDrawLine(arrow1Start, arrow1End);
	pangolin::glDrawLine(arrow2Start, arrow2End);

	glColor3f(1.0f, 0.0f, 0.0f);
	pangolin::glDrawCross(Eigen::Vector2d(*patchExtent_, *patchExtent_), 2);
}

void Visualizer::drawIntegratedNablaOverlay()
{
	pangolin::GlFont::I().Text("Integrated nabla").Draw(1, 1);

	pangolin::GlFont::I().Text("track_id: %d", track_id_).Draw(1, 3);

	glColor3f(1.0f, 0.0f, 0.0f);
	pangolin::glDrawCross(Eigen::Vector2d(*patchExtent_, *patchExtent_), 2);
}

void Visualizer::drawCostMapOverlay()
{
	pangolin::GlFont::I().Text("Cost map").Draw(0, 0);

	pangolin::GlFont::I().Text("track_id: %d", track_id_).Draw(0, 1);
	glColor3f(1.0f, 0.0f, 0.0f);
	pangolin::glDrawCross(Eigen::Vector2d(fmax(0.0, (costMap_.cols - 1) / 2),
										  fmax(0.0, (costMap_.rows - 1) / 2)),
						  0.5);

	glColor3f(1.0f, 0.0f, 0.0f);
	const auto x = initPatch_.x - newPatch_.x;
	const auto y = initPatch_.y - newPatch_.y;
	pangolin::glDrawCross(Eigen::Vector2d((costMap_.cols - 1) / 2 - x,
										  (costMap_.rows - 1) / 2 - y),
						  0.5);
}

void Visualizer::wait() const
{
	std::this_thread::sleep_for(SLEEP_MICROSECONDS);
}

void Visualizer::step()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// wait();

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
	trackerParamsChanged_ = false;
	evaluatorParamsChanged_ = false;
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
	showSettingsPanel_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("ui.showSettings", false, true));

	patchExtent_ = std::unique_ptr<pangolin::Var<int>>(new pangolin::Var<int>(
		"settings.patchExtent", trackerParams_.patchExtent, 0, 50));
	minDistance_ =
		std::unique_ptr<pangolin::Var<double>>(new pangolin::Var<double>(
			"settings.minDistance", trackerParams_.minDistance, 1, 10));
	drawCostMap_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>(
		"settings.drawCostMap", trackerParams_.optimizerParams.drawCostMap,
		true));
	optimizerThreshold_ =
		std::unique_ptr<pangolin::Var<double>>(new pangolin::Var<double>(
			"settings.optimizerThreshold",
			trackerParams_.optimizerParams.optimizerThreshold, 0, 2));
	huberLoss_ =
		std::unique_ptr<pangolin::Var<double>>(new pangolin::Var<double>(
			"settings.huberLoss", trackerParams_.optimizerParams.huberLoss, 0,
			2));

	visualOdometryExperiment_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("settings.visOdometryExperiment",
								evaluatorParams_.visOdometryExperiment, true));

	trackerExperiment_ = std::unique_ptr<pangolin::Var<bool>>(
		new pangolin::Var<bool>("settings.trackerExperiment",
								evaluatorParams_.trackerExperiment, true));
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
	// if (set_)
	// {
	// 	static int id = 0;
	// 	cv::Mat image1;
	// 	cv::cvtColor(originalImage_, image1, cv::COLOR_GRAY2RGB);

	// 	for (const auto& event : integratedEvents_)
	// 	{
	// 		Eigen::Vector2d point(event.value.point.x, event.value.point.y);
	// 		if (event.value.sign == common::EventPolarity::POSITIVE)
	// 		{
	// 			image1.at<cv::Vec3b>(event.value.point.y,
	// 								 event.value.point.x) = {255, 0, 0};
	// 		}
	// 		else
	// 		{  // event.value.sign == common::EventPolarity::NEGATIVE
	// 			image1.at<cv::Vec3b>(event.value.point.y,
	// 								 event.value.point.x) = {0, 0, 255};
	// 		}
	// 	}

	// 	cv::resize(image1, image1, {720, 540});
	// 	for (const auto& patch : patches_)
	// 	{
	// 		if (patch.isLost())
	// 		{
	// 			continue;
	// 		}

	// 		cv::circle(image1, {patch.toCorner().x * 3, patch.toCorner().y * 3},
	// 				   4, {0, 255, 0}, 2);
	// 		const auto& trajectory = patch.getTrajectory();
	// 		for (int i = static_cast<int>(trajectory.size()) - 1;
	// 			 i > static_cast<int>(trajectory.size()) - 60 && i >= 1; --i)
	// 		{
	// 			cv::line(image1,
	// 					 {trajectory[i].value.x * 3, trajectory[i].value.y * 3},
	// 					 {trajectory[i - 1].value.x * 3,
	// 					  trajectory[i - 1].value.y * 3},
	// 					 {127, 0, 127}, 2);
	// 		}
	// 	}

	// 	cv::Mat image2;
	// 	cv::cvtColor(convertImageToGray(integratedNabla_), image2,
	// 				 cv::COLOR_GRAY2RGB);
	// 	cv::resize(image2, image2, {540, 540});

	// 	cv::Mat image3;
	// 	cv::cvtColor(convertImageToGray(predictedNabla_), image3,
	// 				 cv::COLOR_GRAY2RGB);
	// 	cv::resize(image3, image3, {540, 540});
	// 	const auto start =
	// 		Eigen::Vector2d(270 - 55 * std::cos(flow_), 270 - 55 * std::sin(flow_));
	// 	const auto end =
	// 		Eigen::Vector2d(270 + 55 * std::cos(flow_), 270 + 55 * std::sin(flow_));
	// 	const auto arrow1Start =
	// 		Eigen::Vector2d(270 + 55 * std::cos(flow_), 270 + 55 * std::sin(flow_));
	// 	const auto arrow1End = Eigen::Vector2d(270 + 35 * std::cos(flow_ + 0.2),
	// 										   270 + 35 * std::sin(flow_ + 0.2));
	// 	const auto arrow2Start =
	// 		Eigen::Vector2d(270 + 55 * std::cos(flow_), 270 + 55 * std::sin(flow_));
	// 	const auto arrow2End = Eigen::Vector2d(270 + 35 * std::cos(flow_ - 0.2),
	// 										   270 + 35 * std::sin(flow_ - 0.2));
	// 	cv::line(image3, {start(0), start(1)}, {end(0), end(1)}, {50, 0, 127});
	// 	cv::line(image3, {arrow1Start(0), arrow1Start(1)},
	// 			 {arrow1End(0), arrow1End(1)}, {50, 0, 127});
	// 	cv::line(image3, {arrow2Start(0), arrow2Start(1)},
	// 			 {arrow2End(0), arrow2End(1)}, {50, 0, 127});

	// 	cv::Mat grayImage = convertImageToGray(costMap_);
	// 	cv::Mat imColor;
	// 	applyColorMap(grayImage, imColor, cv::COLORMAP_JET);
	// 	cv::Mat image4 = imColor;

	// 	cv::resize(image4, image4, {540, 540});
	// 	cv::hconcat(image1, image2, image1);
	// 	cv::hconcat(image1, image3, image1);
	// 	cv::hconcat(image1, image4, image1);

	// 	cv::imwrite("../results/tracking_images/second_experiment/" +
	// 					std::to_string(id++) + ".png",
	// 				image1);
	// }
	// static int id = 0;
	// cv::imwrite("../results/compensation/integrated" +
	// 					std::to_string(id) + ".png",
	// 				convertImageToGray(integratedNabla_));
	// cv::imwrite("../results/compensation/compensated" +
	// 					std::to_string(id++) + ".png",
	// 				convertImageToGray(compensatedEventImage_));

	pangolin::FinishFrame();
	quit_ = pangolin::ShouldQuit();
	nextPressed_ = pangolin::Pushed(*nextStepButton_);
	nextIntervalPressed_ = pangolin::Pushed(*nextIntervalStepButton_);
	nextImagePressed_ = pangolin::Pushed(*nextImageButton_);

	if (showSettingsPanel_->GuiChanged())
	{
		settingsPanel_->Show(*showSettingsPanel_);
	}

	updateTrackerParams();
	updateEvaluatorParams();
}

void Visualizer::updateTrackerParams()
{
	const auto patchExtent = (*patchExtent_);
	const auto minDistance = (*minDistance_);
	const auto drawCostMap = (*drawCostMap_);
	const auto optimizerThreshold = (*optimizerThreshold_);
	const auto huberLoss = (*huberLoss_);
	if (trackerParams_.patchExtent != patchExtent ||
		trackerParams_.minDistance != minDistance ||
		trackerParams_.optimizerParams.drawCostMap != drawCostMap ||
		trackerParams_.optimizerParams.optimizerThreshold !=
			optimizerThreshold ||
		trackerParams_.optimizerParams.huberLoss != huberLoss)
	{
		trackerParams_.patchExtent = patchExtent;
		trackerParams_.minDistance = minDistance;
		trackerParams_.optimizerParams.drawCostMap = drawCostMap;
		trackerParams_.optimizerParams.optimizerThreshold = optimizerThreshold;
		trackerParams_.optimizerParams.huberLoss = huberLoss;
		trackerParamsChanged_ = true;
	}
}

void Visualizer::updateEvaluatorParams()
{
	const auto visualOdometryExperiment = (*visualOdometryExperiment_);
	const auto trackerExperiment = (*trackerExperiment_);
	if (evaluatorParams_.visOdometryExperiment != visualOdometryExperiment ||
		evaluatorParams_.trackerExperiment != trackerExperiment)
	{
		evaluatorParams_.visOdometryExperiment = visualOdometryExperiment;
		evaluatorParams_.trackerExperiment = trackerExperiment;
		evaluatorParamsChanged_ = true;
	}
}

void Visualizer::setLandmarks(const visual_odometry::MapLandmarks& landmarks)
{
	landmarks_ = landmarks;
}

void Visualizer::setActiveFrames(
	const std::map<size_t, visual_odometry::Keyframe>& frames)
{
	activeFrames_ = frames;
}

void Visualizer::setStoredFrames(
	const std::list<visual_odometry::Keyframe>& frames)
{
	storedFrames_ = frames;
}

void Visualizer::setCompensatedEventImage(const cv::Mat& compensatedEventImage)
{
	compensatedEventImage_ = compensatedEventImage;
}

void Visualizer::setIntegratedEventImage(const cv::Mat& integratedEventImage)
{
	integratedEventImage_ = integratedEventImage;
}

void Visualizer::setGtPoses(const std::vector<common::Pose3d>& gt)
{
	gt_ = gt;
}

void Visualizer::setStoredLandmarks(
	const std::vector<std::pair<tracker::TrackId, Eigen::Vector3d>>& lms)
{
	storedLandmarks_ = lms;
}

}  // namespace tools
