#pragma once

#include "IVideo.h"

namespace imqs {
namespace video {

class NVVideo : public IVideo {
public:
	Error OpenFile(std::string filename) override {
		return Error("NVVideo not available on Windows");
	}
	Error DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds = nullptr) override {
		return Error("NVVideo not available on Windows");
	}
	Error SeekToMicrosecond(int64_t microsecond, unsigned flags = SeekFlagNone) override {
		return Error("NVVideo not available on Windows");
	}
};

} // namespace video
} // namespace imqs