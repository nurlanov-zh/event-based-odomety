#include <dataset_reader/davis240c_reader.h>
#include <replayer/replayer.h>
#include "visualizer/visualizer.h"

#include <CLI/CLI.hpp>

#include <thread>

tools::Visualizer visualizer;
// temporary callback

cv::Mat image;
bool received = false;
common::Corners defaultCorners = {{{5, 5}, 0}, {{25, 25}, 1.57}};

void newImageCallback(const common::Sample<cv::Mat>& sample)
{
	image	= sample.value;
	received = true;
}

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

	std::shared_ptr<tools::DatasetReader> reader =
		std::make_shared<tools::Davis240cReader>(dataset);

	tools::Replayer replayer(reader);

	if (showGui) {
		visualizer.createWindow();
		replayer.addImageCallback(newImageCallback);
	}

	std::thread thread([&replayer, showGui]() {
		while (!replayer.finished() && !visualizer.shouldQuit()) {
			const bool stop = visualizer.stopPressed();
			if (!stop || !showGui) {
				replayer.next();
			}
			else if (stop && showGui)
			{
				if (visualizer.nextPressed()) {
					replayer.next();
				}
				else if (visualizer.nextIntervalPressed())
				{
					replayer.nextInterval(visualizer.getStepInterval());
				}
			}
		}
	});

	if (showGui) {
		while (!visualizer.shouldQuit()) {
			if (received) {
				visualizer.drawOriginalImage(image);
				visualizer.drawPredictedFlow(image);
				visualizer.drawIntegratedFlow(image);
                visualizer.setCorners(defaultCorners);
            }
            visualizer.setTimestamp(replayer.getLastTimestamp());
            visualizer.step();
		}
	}

	thread.join();

	return 0;
}