#pragma once
#include <common/data_types.h>
#include <dataset_reader/dataset_reader.h>

#include <functional>
#include <iterator>
#include <memory>
#include <string>

namespace tools
{
#define REGISTER_CALLBACK(class, callback, object)                             \
	std::bind(&class::callback, &object, std::placeholders::_1)

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

	bool finished() const;

	void addGroundTruthCallback(
		std::function<void(const common::Sample<common::Pose3d>&)> callback);

	void addEventCallback(
		std::function<void(const common::Sample<common::Event>&)> callback);

	void addImageCallback(
		std::function<void(const common::Sample<cv::Mat>&)> callback);

	void next();

    void nextInterval(const common::timestamp_t& interval);

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

	std::vector<std::function<void(const common::Sample<common::Pose3d>&)>>
		groundTruthCallbacks_;
	std::vector<std::function<void(const common::Sample<common::Event>&)>>
		eventCallbacks_;
	std::vector<std::function<void(const common::Sample<cv::Mat>&)>>
		imageCallbacks_;
};

}  // ns tools