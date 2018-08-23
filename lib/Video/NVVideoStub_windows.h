#pragma once

namespace imqs {
namespace video {

class NVVideo : public IVideo {
public:
	Error OpenFile(std::string filename) override {
		return Error("Not available on Windows");
	}
	Error DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds = nullptr) override {
		return Error("Not available on Windows");
	}
};

} // namespace video
} // namespace imqs