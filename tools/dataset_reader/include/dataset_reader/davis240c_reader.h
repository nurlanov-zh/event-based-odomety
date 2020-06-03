#pragma once

#include "dataset_reader.h"

namespace tools
{
class Davis240cReader : public DatasetReader
{
   public:
	Davis240cReader(const std::string& path);

	common::EventSequence getEvents() const override;

	common::ImageSequence getImages() const override;

	common::GroundTruth getGroundTruth() const override;

	common::EventSample getEventSample(std::string& line) const;
	common::ImageSample getImageSample(std::string& line) const;
	common::GroundTruthSample getGroundTruthSample(std::string& line) const;
};

}  // namespace tools