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
	EVENT = 0,
	IMAGE = 1,
	GROUND_TRUTH = 2
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

	const common::GroundTruth& getGroundTruth() const { return groundTruth_; }

	const tracker::Patches& getPatches() const { return traj_; }

   private:
	template <typename T>
	void notify(const std::vector<std::function<void(const T&)>>& callbacks,
				const T& message)
	{
		for (const auto& callback : callbacks)
		{
			callback(message);
		}
	}

   private:
	std::shared_ptr<spdlog::logger> consoleLog_;
	std::shared_ptr<spdlog::logger> errLog_;

	std::shared_ptr<DatasetReader> reader_;

	common::GroundTruth groundTruth_;
	common::EventSequence events_;
	common::ImageSequence images_;

	bool hasEvents_;

	common::EventSequence::iterator eventIt_;
	common::ImageSequence::iterator imageIt_;

	common::timestamp_t lastTimestamp_;

	tracker::Patches traj_;

	bool imageArrived_;

	std::vector<std::function<void(const common::GroundTruthSample&)>>
		groundTruthCallbacks_;
	std::vector<std::function<void(const common::EventSample&)>>
		eventCallbacks_;
	std::vector<std::function<void(const common::ImageSample&)>>
		imageCallbacks_;
};

}  // namespace tools