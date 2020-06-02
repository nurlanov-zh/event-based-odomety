#pragma once
#include <common/data_types.h>
#include <dataset_reader/dataset_reader.h>

#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#include <functional>
#include <iterator>
#include <memory>
#include <string>

#define REGISTER_CALLBACK(class, callback, object)                             \
	std::bind(&class ::callback, &object, std::placeholders::_1)

namespace tools
{
enum EventType
{
	EVENT		 = 0,
	IMAGE		 = 1,
	GROUND_TRUTH = 2
};

struct TimestampCompare
{
	bool operator()(const std::pair<common::timestamp_t, EventType>& lhs,
					const std::pair<common::timestamp_t, EventType>& rhs)
	{
		return lhs.first > rhs.first;
	}
};

class Replayer
{
   public:
	Replayer(const std::shared_ptr<DatasetReader> reader);

	void reset();

	bool finished() const;

	void addGroundTruthCallback(
		std::function<void(const common::GroundTruthSample&)> callback);

	void addEventCallback(
		std::function<void(const common::EventSample&)> callback);

	void addImageCallback(
		std::function<void(const common::ImageSample&)> callback);

	void next();

	void nextInterval(const common::timestamp_t& interval);

	void nextImage();

	common::timestamp_t getLastTimestamp() { return lastTimestamp_; }

   private:
	template <typename T>
	void notify(const std::vector<std::function<void(const T&)>>& callbacks,
				const T& message)
	{
		for (const auto& callback : callbacks) {
			callback(message);
		}
	}

	template <typename T>
	void fillTimestampsQueue(const T& sequence, const EventType& type)
	{
		for (const auto& sample : sequence) {
			timestampsQueue_.push(std::make_pair(sample.timestamp, type));
		}
	}

   private:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	common::GroundTruth groundTruth_;
	common::EventSequence events_;
	common::ImageSequence images_;
	std::priority_queue<std::pair<common::timestamp_t, EventType>,
						std::vector<std::pair<common::timestamp_t, EventType>>,
						TimestampCompare>
		timestampsQueue_;

	common::EventSequence::iterator eventIt_;
	common::ImageSequence::iterator imageIt_;

	common::timestamp_t lastTimestamp_;

	bool imageArrived_;

	std::vector<std::function<void(const common::GroundTruthSample&)>>
		groundTruthCallbacks_;
	std::vector<std::function<void(const common::EventSample&)>>
		eventCallbacks_;
	std::vector<std::function<void(const common::ImageSample&)>>
		imageCallbacks_;
};

}  // ns tools