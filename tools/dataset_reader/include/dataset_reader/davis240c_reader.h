#pragma once

#include "dataset_reader.h"

namespace tools
{
class Davis240cReader : DatasetReader
{
   public:
	Davis240cReader(const std::string& path);

	common::EventSequence getEvents() const override;

	common::ImageSequence getImages() const override;

	common::GroundTruth getGroundTruth() const override;
};

}  // namespace tools