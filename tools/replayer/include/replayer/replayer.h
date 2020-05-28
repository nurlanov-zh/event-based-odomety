#pragma once
#include <common/data_types.h>

#include <memory>
#include <string>

namespace tools
{
class Replayer
{
   public:
	Replayer(const std::string& path);

    bool finished() const;

    void wait() const;

    common::Sample<cv::Mat> getCurrentImage();

    void next();

   private:
	common::GroundTruth groundTruth_;
	common::EventSequence events_;
	common::ImageSequence images_;
    size_t currentImageIdx_ = 0;
};

}  // ns tools