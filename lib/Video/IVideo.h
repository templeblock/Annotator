#pragma once

namespace imqs {
namespace video {

class IVideo {
public:
	virtual Error OpenFile(std::string filename)                                                               = 0;
	virtual Error DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds = nullptr) = 0;
};

} // namespace video
} // namespace imqs