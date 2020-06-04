#include <dataset_reader/davis240c_reader.h>
#include <evaluator/evaluator.h>
#include <replayer/replayer.h>
#include "visualizer/visualizer.h"

#include <CLI/CLI.hpp>

#include <thread>

tools::Visualizer visualizer;
const std::chrono::microseconds REDRAW_DELAY_MICROSECONDS =
	std::chrono::microseconds(5000);

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

	const auto param = tools::EvaluatorParams();
	tools::Evaluator evaluator(param);

	replayer.addEventCallback(
		REGISTER_CALLBACK(tools::Evaluator, eventCallback, evaluator));
	replayer.addImageCallback(
		REGISTER_CALLBACK(tools::Evaluator, imageCallback, evaluator));

	replayer.addEventCallback(
		REGISTER_CALLBACK(tools::Visualizer, eventCallback, visualizer));
	replayer.addImageCallback(
		REGISTER_CALLBACK(tools::Visualizer, imageCallback, visualizer));

	if (showGui) {
		visualizer.createWindow();
	}

	while (!visualizer.shouldQuit() && !replayer.finished()) {
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
			else if (visualizer.nextImagePressed())
			{
				replayer.nextImage();
			}
		}

		if (visualizer.resetPressed()) {
			replayer.reset();
			evaluator.reset();
		}

		if (showGui) {
			const auto timestamp = replayer.getLastTimestamp();

			// redraw only every so often if not stop
			if (stop ||
				(!stop &&
				 (timestamp.count() % REDRAW_DELAY_MICROSECONDS.count() == 0)))
			{
				visualizer.setPatches(evaluator.getPatches());
				visualizer.setTimestamp(timestamp);
				visualizer.step();
			}
		}
	}

	return 0;
}