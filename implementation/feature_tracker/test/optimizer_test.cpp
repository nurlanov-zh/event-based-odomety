#include <feature_tracker/optimizer.h>

#include <opencv2/core/eigen.hpp>

#include <gtest/gtest.h>

std::string TEST_DATA_PATH = "test/test_data";
bool useRotation = false;

class LoggingTest
{
   public:
	LoggingTest()
	{
		spdlog::stdout_color_mt("console");
		spdlog::stderr_color_mt("stderr");
	}
};

LoggingTest logger;

void saveImage(cv::Mat image, std::string name, bool isCostMap = false)
{
	cv::Mat grayImage;
	double minVal;
	double maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);
	image.convertTo(grayImage, CV_8U, 255.0 / (maxVal - minVal),
					-minVal * 255.0 / (maxVal - minVal));
	if (isCostMap)
	{
		cv::Mat imColor;
		applyColorMap(grayImage, imColor, cv::COLORMAP_JET);
		grayImage = imColor;
	}
	cv::imwrite("/tmp/" +
					std::string(::testing::UnitTest::GetInstance()
									->current_test_info()
									->name()) +
					"_" + name + ".png",
				grayImage);
}

cv::Mat warpGeneral(const common::Pose2d& warp, double flowDir,
					const cv::Mat& gradX, const cv::Mat& gradY)
{
	cv::Mat warpedGradX;
	cv::Mat warpedGradY;
	cv::Mat warpCv;
	cv::eigen2cv(warp.matrix2x3(), warpCv);

	const auto center =
		Eigen::Vector2d((gradX.cols - 1) / 2, (gradX.rows - 1) / 2);
	const Eigen::Vector2d offsetToCenter =
		-(warp.rotationMatrix() * center) + center;
	warpCv.at<double>(0, 2) += offsetToCenter.x();
	warpCv.at<double>(1, 2) += offsetToCenter.y();

	cv::warpAffine(gradX, warpedGradX, warpCv, {gradX.cols, gradX.rows},
				   cv::INTER_CUBIC);
	cv::warpAffine(gradY, warpedGradY, warpCv, {gradY.cols, gradY.rows},
				   cv::INTER_CUBIC);

	auto warpedNabla =
		-warpedGradX * std::cos(flowDir) - warpedGradY * std::sin(flowDir);
	return warpedNabla;
}

TEST(Optimizer, optimizerSimpleTest)
{
	const cv::Size2i imageSize = {35, 35};
	tracker::OptimizerParams params;
	params.drawCostMap = true;
	tracker::Optimizer optimizer(params, imageSize);

	const auto randomInt = [](int min, int max) -> int {
		return min + std::rand() % (max - min);
	};

	for (size_t i = 0; i < 5; ++i)
	{
		cv::Mat gradX = cv::Mat::zeros(imageSize, CV_64F);
		cv::Mat gradY = cv::Mat::zeros(imageSize, CV_64F);

		// around the middle of the patch
		cv::line(gradX, {randomInt(0, imageSize.width), imageSize.width / 2},
				 {randomInt(0, imageSize.width), imageSize.width / 2}, 2);
		cv::line(gradY, {imageSize.height / 2, randomInt(0, imageSize.height)},
				 {imageSize.height / 2, randomInt(0, imageSize.height)}, 2);

		double rotation =
			useRotation ? M_PI / static_cast<double>(randomInt(24, 30)) : 0;
		common::Pose2d warp(
			rotation, Eigen::Vector2d(randomInt(-5, 5), randomInt(-5, 5)));

		const Eigen::Vector2d shift =
			Eigen::Vector2d(randomInt(-3, 3), randomInt(-3, 3));
		common::Pose2d warpInit(0, shift);
		const double flowDir = std::atan2(shift.y(), shift.x());

		// Center of the patch is important! Because the warping is done around
		// it.
		tracker::Patch patch({17, 17}, 17, common::timestamp_t(0));
		patch.setWarp(warpInit);
		patch.setFlowDir(flowDir);

		const auto integratedNabla =
			warpGeneral(warp.inverse(), flowDir, gradX, gradY);
		patch.setIntegratedNabla(integratedNabla);

		cv::Mat gradXBlurred;
		cv::Mat gradYBlurred;
		cv::GaussianBlur(gradX, gradXBlurred, cv::Size(9, 9), 0, 0);
		cv::GaussianBlur(gradY, gradYBlurred, cv::Size(9, 9), 0, 0);

		optimizer.setGrad(gradXBlurred, gradYBlurred);
		optimizer.optimize(patch);

		const auto flowDirOut = patch.getFlow();
		const auto warpOut = patch.getWarp();

		auto predictedNabla =
			warpGeneral(warpOut.inverse(), flowDirOut, gradX, gradY);

		EXPECT_NEAR(flowDirOut, flowDir, 5e-1);
		auto warpOutTangent = warpOut.log();
		auto warpTangent = warp.log();
		EXPECT_NEAR(warpOutTangent.x(), warpTangent.x(), 5e-1);
		EXPECT_NEAR(warpOutTangent.y(), warpTangent.y(), 5e-1);
		EXPECT_NEAR(warpOutTangent.z(), warpTangent.z(), 5e-1);

		if (std::abs(flowDir - flowDirOut) > 1e-1 or
			std::abs(warpTangent.x() - warpOutTangent.x()) > 5e-1 or
			std::abs(warpTangent.y() - warpOutTangent.y()) > 5e-1 or
			std::abs(warpTangent.z() - warpOutTangent.z()) > 1e-1)
		{
			saveImage(patch.getCostMap(), "costMap_" + std::to_string(i), true);
		}

#ifdef SAVE_IMAGES
		saveImage(gradX, "gradX" + std::to_string(i));
		saveImage(gradY, "gradY" + std::to_string(i));
		saveImage(integratedNabla / norm(integratedNabla),
				  "integratedNabla" + std::to_string(i));
		saveImage(predictedNabla / norm(predictedNabla),
				  "predictedNabla" + std::to_string(i));
#endif
	}
}