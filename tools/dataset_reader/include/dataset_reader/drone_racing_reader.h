#pragma once

#include "dataset_reader.h"

namespace tools
{
class DroneRacingReader : public DatasetReader
{
   public:
	DroneRacingReader(const std::string& path);

	std::optional<common::EventSequence> getEvents() override;

	common::ImageSequence getImages() const override;

	common::GroundTruth getGroundTruth() const override;

	common::CameraModelParams<double> getCalibration() const override;

	tracker::Patches getTrajectory() const override;

	common::EventSample getEventSample(std::string& line) const;
	common::ImageSample getImageSample(std::string& line) const;
	common::GroundTruthSample getGroundTruthSample(std::string& line) const;
	common::CameraModelParams<double> getCalibrationLine(
		std::string& line) const;
	tracker::Patch getTrajectoryLine(std::string& line) const;
};

}  // namespace tools