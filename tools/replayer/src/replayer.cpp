#include "replayer/replayer.h"

namespace tools
{
Replayer::Replayer(const std::shared_ptr<DatasetReader> reader): lastTimestamp_(0)
{
	images_		 = reader->getImages();
	events_		 = reader->getEvents();
	groundTruth_ = reader->getGroundTruth();

	fillTimestampsQueue(images_, EventType::IMAGE);
	fillTimestampsQueue(events_, EventType::EVENT);
	fillTimestampsQueue(groundTruth_, EventType::GROUND_TRUTH);

	eventIt_ = events_.begin();
	imageIt_ = images_.begin();
}

bool Replayer::finished() const
{
	return timestampsQueue_.empty();
}

void Replayer::next()
{
	const auto minSample = timestampsQueue_.top();
	timestampsQueue_.pop();
	switch (minSample.second)
	{
		case EventType::EVENT:
			notify(eventCallbacks_, *eventIt_);
			if (eventIt_ != events_.end()) {
				++eventIt_;
			}
			break;

		case EventType::IMAGE:
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
    if (finished())
    {
        return;
    }

    next();
    const auto firstTime = lastTimestamp_;
    auto lastTime = firstTime;
    do {
        next();
    } while((lastTimestamp_ - firstTime) < interval && !finished());
}

void Replayer::addGroundTruthCallback(
	std::function<void(const common::Sample<common::Pose3d>&)> callback)
{
	groundTruthCallbacks_.push_back(callback);
}

void Replayer::addEventCallback(
	std::function<void(const common::Sample<common::Event>&)> callback)
{
	eventCallbacks_.push_back(callback);
}

void Replayer::addImageCallback(
	std::function<void(const common::Sample<cv::Mat>&)> callback)
{
	imageCallbacks_.push_back(callback);
}

}  // ns tools