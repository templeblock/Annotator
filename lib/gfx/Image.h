#pragma once

namespace imqs {
namespace gfx {

enum class ImageFormat {
	Null,
	Png,
	Jpeg,
};

const char* ImageFormatName(ImageFormat f);
ImageFormat ParseImageFormat(const std::string& f);

} // namespace gfx
} // namespace imqs