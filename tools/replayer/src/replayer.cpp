#include "replayer/replayer.h"

namespace tools
{
Replayer::Replayer(const std::shared_ptr<DatasetReader> reader)
	: reader_(reader), lastTimestamp_(0)
{
	hasEvents_ = false;
	images_ = reader->getImages();

	const auto& events = reader->getEvents();
	if (events.has_value())
	{
		hasEvents_ = true;
		events_ = events.value();
	}

	groundTruth_ = reader->getGroundTruth();
	
	traj_ = reader->getTrajectory();

	reset();

	consoleLog_ = spdlog::get("console");
	errLog_ = spdlog::get("stderr");
}

bool Replayer::finished() const
{
	return !hasEvents_ || imageIt_ == images_.end();
}

void Replayer::reset()
{
	eventIt_ = events_.begin();
	imageIt_ = images_.begin();

	lastTimestamp_ = common::timestamp_t(0);
	imageArrived_ = false;
}

void Replayer::next()
{
	if (hasEvents_ && eventIt_ == events_.end())
	{
		const auto& events = reader_->getEvents();
		if (events.has_value())
		{
			hasEvents_ = true;
			events_ = events.value();
			eventIt_ = events_.begin();
		}
	}

	if (eventIt_->timestamp < imageIt_->timestamp)
	{
		lastTimestamp_ = eventIt_->timestamp;
		consoleLog_->trace("New event sample is arrived at time {:08d}",
						   imageIt_->timestamp.count());

		notify(eventCallbacks_, *eventIt_);
		if (eventIt_ != events_.end())
		{
			++eventIt_;
		}
	}
	else
	{
		lastTimestamp_ = imageIt_->timestamp;
		consoleLog_->trace("New image sample is arrived at time {:08d}",
						   imageIt_->timestamp.count());

		imageArrived_ = true;
		notify(imageCallbacks_, *imageIt_);
		if (imageIt_ != images_.end())
		{
			++imageIt_;
		}
	}
}

void Replayer::nextInterval(const common::timestamp_t& interval)
{
	if (finished())
	{
		return;
	}

	next();
	const auto firstTime = lastTimestamp_;
	auto lastTime = firstTime;
	do
	{
		next();
	} while ((lastTimestamp_ - firstTime) < interval && !finished());
}

void Replayer::nextImage()
{
	if (finished())
	{
		return;
	}
	imageArrived_ = false;

	while (!imageArrived_)
	{
		next();
	}
}

void Replayer::addGroundTruthCallback(
	std::function<void(const common::GroundTruthSample&)> callback)
{
	consoleLog_->debug("New ground truth callback is registered");
	groundTruthCallbacks_.push_back(callback);
}

void Replayer::addEventCallback(
	std::function<void(const common::EventSample&)> callback)
{
	consoleLog_->debug("New event callback is registered");
	eventCallbacks_.push_back(callback);
}

void Replayer::addImageCallback(
	std::function<void(const common::ImageSample&)> callback)
{
	consoleLog_->debug("New image callback is registered");
	imageCallbacks_.push_back(callback);
}

}  // namespace tools