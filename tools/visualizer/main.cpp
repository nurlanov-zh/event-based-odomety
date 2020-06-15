#include <dataset_reader/davis240c_reader.h>
#include <evaluator/evaluator.h>
#include <replayer/replayer.h>
#include "visualizer/visualizer.h"

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>

#include <thread>

tools::Visualizer visualizer;
const std::chrono::microseconds REDRAW_DELAY_MICROSECONDS =
	std::chrono::microseconds(500);

int main(int argc, char** argv)
{
	spdlog::set_level(spdlog::level::from_str("info"));

	spdlog::stdout_color_mt("console");
	spdlog::stderr_color_mt("stderr");
	auto console = spdlog::get("console");
	auto errLogger = spdlog::get("stderr");

	console->info("Event based odometry has been started!");

	bool showGui = true;
	std::string dataset = "../data/DAVIS240C/shapes_rotation";

	CLI::App app{"Event based odometry"};

	app.add_option("--show-gui", showGui, "Show GUI");
	app.add_option("--dataset", dataset, "Dataset. Default: " + dataset);

	console->info("Options passed are:");
	console->info("\tshow-gui: {}", showGui);
	console->info("\tdataset: {}", dataset);

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

	tools::EvaluatorParams param;
	param.drawImages = showGui;
	tools::Evaluator evaluator(param);

	replayer.addEventCallback(
		REGISTER_CALLBACK(tools::Evaluator, eventCallback, evaluator));
	replayer.addImageCallback(
		REGISTER_CALLBACK(tools::Evaluator, imageCallback, evaluator));

	if (showGui)
	{
		replayer.addEventCallback(
			REGISTER_CALLBACK(tools::Visualizer, eventCallback, visualizer));
		replayer.addImageCallback(
			REGISTER_CALLBACK(tools::Visualizer, imageCallback, visualizer));
		visualizer.createWindow();
	}

	while (!visualizer.shouldQuit() && !replayer.finished())
	{
		const bool stop = visualizer.stopPressed();
		if (!stop || !showGui)
		{
			replayer.next();
		}
		else if (stop && showGui)
		{
			if (visualizer.nextPressed())
			{
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

		if (showGui)
		{
			const auto timestamp = replayer.getLastTimestamp();

			// redraw only every so often if not stop
			if (stop ||
				(!stop &&
				 (timestamp.count() % REDRAW_DELAY_MICROSECONDS.count() == 0)))
			{
				visualizer.setPatches(evaluator.getPatches());
				visualizer.setTimestamp(timestamp);
				visualizer.step();

				if (visualizer.isTrackerParamsChanged())
				{
					auto trackerParams = visualizer.getTrackerParams();
					trackerParams.drawImages = true;
					evaluator.setTrackerParams(trackerParams);
				}
			}
		}
	}

	return 0;
}