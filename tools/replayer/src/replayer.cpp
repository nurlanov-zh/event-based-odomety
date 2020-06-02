#include "replayer/replayer.h"

namespace tools
{
Replayer::Replayer(const std::shared_ptr<DatasetReader> reader)
	: lastTimestamp_(0)
{
	images_		 = reader->getImages();
	events_		 = reader->getEvents();
	groundTruth_ = reader->getGroundTruth();

	reset();

	consoleLog_ = spdlog::get("console");
	errLog_		= spdlog::get("stderr");
}

bool Replayer::finished() const
{
	return timestampsQueue_.empty();
}

void Replayer::reset()
{
	while (!timestampsQueue_.empty()) {
		timestampsQueue_.pop();
	}

	fillTimestampsQueue(images_, EventType::IMAGE);
	fillTimestampsQueue(events_, EventType::EVENT);
	fillTimestampsQueue(groundTruth_, EventType::GROUND_TRUTH);

	eventIt_ = events_.begin();
	imageIt_ = images_.begin();

	lastTimestamp_ = common::timestamp_t(0);
	imageArrived_  = false;
}

void Replayer::next()
{
	const auto minSample = timestampsQueue_.top();
	timestampsQueue_.pop();
	switch (minSample.second)
	{
		case EventType::EVENT:

			consoleLog_->trace("New event sample is arrived at time {:08d}",
							   imageIt_->timestamp.count());

			notify(eventCallbacks_, *eventIt_);
			if (eventIt_ != events_.end()) {
				++eventIt_;
			}
			break;

		case EventType::IMAGE:

			consoleLog_->trace("New image sample is arrived at time {:08d}",
							   imageIt_->timestamp.count());

			imageArrived_ = true;
			notify(imageCallbacks_, *imageIt_);
			if (imageIt_ != images_.end()) {
				++imageIt_;
			}
			break;

		case EventType::GROUND_TRUTH:
			break;
	}
	lastTimestamp_ = minSample.first;
}

void Replayer::nextInterval(const common::timestamp_t& interval)
{
	if (finished()) {
		return;
	}

	next();
	const auto firstTime = lastTimestamp_;
	auto lastTime		 = firstTime;
	do
	{
		next();
	} while ((lastTimestamp_ - firstTime) < interval && !finished());
}

void Replayer::nextImage()
{
	if (finished()) {
		return;
	}
	imageArrived_ = false;

	while (!imageArrived_) {
		next();
	}
}

void Replayer::addGroundTruthCallback(
	std::function<void(const common::GroundTruthSample&)> callback)
{
	spdlog::get("console")->debug("New ground truth callback is registered");
	groundTruthCallbacks_.push_back(callback);
}

void Replayer::addEventCallback(
	std::function<void(const common::EventSample&)> callback)
{
	spdlog::get("console")->debug("New event callback is registered");
	eventCallbacks_.push_back(callback);
}

void Replayer::addImageCallback(
	std::function<void(const common::ImageSample&)> callback)
{
	spdlog::get("console")->debug("New image callback is registered");
	imageCallbacks_.push_back(callback);
}

}  // ns tools