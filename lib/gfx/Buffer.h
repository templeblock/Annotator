#pragma once

namespace imqs {
namespace gfx {

class Buffer {
public:
	uint8_t* Data   = nullptr;
	int      Stride = 0;
	int      Width  = 0;
	int      Height = 0;

	Buffer() {}
	Buffer(void* data, int stride, int width, int height) : Data((uint8_t*) data), Stride(stride), Width(width), Height(height) {}
};

} // namespace gfx
} // namespace imqs
