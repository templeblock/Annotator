#pragma once

namespace imqs {
namespace roadproc {

enum class SpeedOutputMode {
	CSV,
	JSON,
};

Error DoSpeed(std::vector<std::string> videoFiles, float zy, double startTime, SpeedOutputMode outputMode, std::string outputFile);

} // namespace roadproc
} // namespace imqs