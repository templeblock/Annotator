#pragma once

#include "Perspective.h"

namespace imqs {
namespace roadproc {

enum class SpeedOutputMode {
	CSV,
	JSON,
};

Error DoSpeed(std::vector<std::string> videoFiles, FlattenParams fp, double startTime, SpeedOutputMode outputMode, std::string outputFile);

} // namespace roadproc
} // namespace imqs