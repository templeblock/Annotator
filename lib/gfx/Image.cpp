#include "pch.h"
#include "Image.h"

using namespace std;

namespace imqs {
namespace gfx {

const char* ImageTypeName(ImageType f) {
	switch (f) {
	case ImageType::Null: return "null";
	case ImageType::Jpeg: return "jpeg";
	case ImageType::Png: return "png";
	}
	return "";
}

ImageType ParseImageType(const std::string& f) {
	if (strings::eqnocase(f, "jpeg") || strings::eqnocase(f, "jpg"))
		return ImageType::Jpeg;
	else if (strings::eqnocase(f, "png"))
		return ImageType::Png;
	else
		return ImageType::Null;
}

Image::Image(ImageFormat format, ContructMode mode, int stride, void* data, int width, int height) {
	if (mode == ConstructCopy) {
		Alloc(format, width, height, stride);
		uint8_t* src = (uint8_t*) data;
		for (int y = 0; y < height; y++) {
			memcpy(Line(y), src + (size_t) y * (size_t) stride, BytesPerPixel() * width);
		}
	} else if (mode == ConstructWindow || mode == ConstructTakeOwnership) {
		IMQS_ASSERT(stride != 0);
		Format  = format;
		Data    = (uint8_t*) data;
		Width   = width;
		Height  = height;
		Stride  = stride;
		OwnData = mode == ConstructTakeOwnership;
	}
}

Image::Image(ImageFormat format, int width, int height, int stride) {
	Alloc(format, width, height, stride);
}

Image::Image(const Image& b) {
	*this = b;
}

Image::Image(Image&& b) {
	*this = std::move(b);
}

Image::~Image() {
	if (OwnData)
		free(Data);
}

Image& Image::operator=(const Image& b) {
	Reset();
	Width   = b.Width;
	Height  = b.Height;
	Stride  = b.Stride;
	Data    = (uint8_t*) imqs_malloc_or_die(Height * Stride);
	Format  = b.Format;
	OwnData = true;
	for (int y = 0; y < Height; y++)
		memcpy(Line(y), b.Line(y), BytesPerLine());
	return *this;
}

Image& Image::operator=(Image&& b) {
	if (this != &b) {
		Reset();
		memcpy(this, &b, sizeof(b));
		memset(&b, 0, sizeof(b));
	}
	return *this;
}

void Image::Reset() {
	if (OwnData)
		free(Data);
	Width   = 0;
	Height  = 0;
	Stride  = 0;
	Data    = nullptr;
	OwnData = false;
	Format  = ImageFormat::Null;
}

void Image::Alloc(ImageFormat format, int width, int height, int stride) {
	if (stride == 0)
		stride = gfx::BytesPerPixel(format) * width;

	IMQS_ASSERT(stride >= gfx::BytesPerPixel(format) * width);

	if (Width == width && Height == height && Stride == stride) {
		Format = format;
		return;
	}

	Reset();

	OwnData = true;
	Width   = width;
	Height  = height;
	Stride  = stride;
	Format  = format;
	Data    = (uint8_t*) imqs_malloc_or_die(Height * std::abs(Stride));
}

Image Image::Window(int x, int y, int width, int height) const {
	IMQS_ASSERT(width >= 0);
	IMQS_ASSERT(height >= 0);
	IMQS_ASSERT(x >= 0);
	IMQS_ASSERT(y >= 0);
	IMQS_ASSERT(x + width <= Width);
	IMQS_ASSERT(y + height <= Height);
	return Image(Format, ConstructWindow, Stride, const_cast<uint8_t*>(At(x, y)), width, height);
}

} // namespace gfx
} // namespace imqs
