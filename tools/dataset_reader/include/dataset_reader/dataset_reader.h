#pragma once

#include "common/data_types.h"

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <string>

namespace tools
{
class DatasetReader
{
   public:
	explicit DatasetReader(const std::string& path) : path_(path) {}

	virtual common::EventSequence getEvents() const = 0;

	virtual common::ImageSequence getImages() const = 0;

	virtual common::GroundTruth getGroundTruth() const = 0;

   protected:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	std::string path_;
};

}  // namespace tools