#include "replayer/replayer.h"

#include <dataset_reader/davis240c_reader.h>

#include <chrono>
#include <thread>

namespace tools
{
const std::chrono::milliseconds SLEEP_MILLISECONDS = std::chrono::milliseconds(5);

Replayer::Replayer(const std::string& path)
{
	const auto reader = Davis240cReader(path);
	images_			  = reader.getImages();
	events_			  = reader.getEvents();
	groundTruth_	  = reader.getGroundTruth();
}

bool Replayer::finished() const
{
	return false;
}

void Replayer::wait() const
{
    std::this_thread::sleep_for(SLEEP_MILLISECONDS);
}

common::Sample<cv::Mat> Replayer::getCurrentImage()
{
    const auto idx = std::min(images_.size() - 1, currentImageIdx_);
    return images_[idx];
}

void Replayer::next()
{
    if (currentImageIdx_ < images_.size())
    {
        currentImageIdx_++;
    }
}


}  // ns tools