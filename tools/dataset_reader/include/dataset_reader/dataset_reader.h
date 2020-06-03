#pragma once

#include "common/data_types.h"
#include "dataset_reader/mapped_file.h"

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <string>

namespace tools
{
const size_t NUM_THREADS = std::thread::hardware_concurrency();

class DatasetReader
{
   public:
	explicit DatasetReader(const std::string& path) : path_(path) {}

	virtual common::EventSequence getEvents() const = 0;

	virtual common::ImageSequence getImages() const = 0;

	virtual common::GroundTruth getGroundTruth() const = 0;

   protected:
	template <typename T, typename D>
	T readFile(const std::string& path,
			   std::function<D(std::string&)> getData) const
	{
		T sequence;
		MappedFile mappedFile(path);
		const char* filePtr = mappedFile.begin();
		const char* lineTs  = nullptr;

		std::vector<std::string> lines;
		while ((lineTs = strchr(filePtr, '\n'))) {
			lines.push_back(std::string(filePtr, lineTs));
			filePtr = lineTs + 1;
		}

		std::vector<T> sequenceThreads(NUM_THREADS);
		std::vector<size_t> startLines;
		std::vector<size_t> linesPerThread;

		for (size_t i = 0; i < NUM_THREADS - 1; ++i) {
			startLines.push_back(i * (lines.size() / NUM_THREADS));
			linesPerThread.push_back(lines.size() / NUM_THREADS);
		}

		startLines.push_back((NUM_THREADS - 1) * (lines.size() / NUM_THREADS));
		linesPerThread.push_back(
			lines.size() - (NUM_THREADS - 1) * (lines.size() / NUM_THREADS));

		std::vector<std::thread> threads;
		for (size_t threadIdx = 0; threadIdx < NUM_THREADS; ++threadIdx) {
			threads.push_back(
				std::thread([&lines, &sequenceThreads, &linesPerThread,
							 &startLines, threadIdx, getData] {
					for (size_t idx = 0; idx < linesPerThread[threadIdx]; ++idx)
					{
						auto line = lines[startLines[threadIdx] + idx];
						sequenceThreads[threadIdx].emplace_back(getData(line));
					}
				}));
		}

		for (size_t threadIdx = 0; threadIdx < NUM_THREADS; ++threadIdx) {
			threads[threadIdx].join();
		}

		sequenceThreads.reserve(lines.size());
		for (auto& items : sequenceThreads) {
			std::move(items.begin(), items.end(), std::back_inserter(sequence));
		}

		return sequence;
	}

   protected:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	std::string path_;
};

}  // namespace tools