#include "pch.h"
#include "Image.h"

namespace imqs {
namespace gfx {

const char* ImageFormatName(ImageFormat f) {
	switch (f) {
	case ImageFormat::Null: return "null";
	case ImageFormat::Jpeg: return "jpeg";
	case ImageFormat::Png: return "png";
	}
	return "";
}

ImageFormat ParseImageFormat(const std::string& f) {
	if (strings::eqnocase(f, "jpeg") || strings::eqnocase(f, "jpg"))
		return ImageFormat::Jpeg;
	else if (strings::eqnocase(f, "png"))
		return ImageFormat::Png;
	else
		return ImageFormat::Null;
}

} // namespace gfx
} // namespace imqs
