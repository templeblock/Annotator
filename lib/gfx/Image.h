#pragma once

#include "Rect.h"
#include "Color8.h"

namespace imqs {
namespace gfx {

enum class ImageType {
	Null,
	Png,
	Jpeg,
};

const char* ImageTypeName(ImageType f);
ImageType   ParseImageType(const std::string& f);

enum class ImageFormat {
	Null,
	RGBA,  // RGBA not premultipled
	RGBAP, // RGBA premultipled
	Gray,
};

inline int BytesPerPixel(ImageFormat f) {
	switch (f) {
	case ImageFormat::Null: return 0;
	case ImageFormat::RGBA: return 4;
	case ImageFormat::RGBAP: return 4;
	case ImageFormat::Gray: return 1;
	}
	return 0;
}

// An image in memory.
// The buffer is by default owned by the Image object, but it doesn't need to be.
// There is no reference counting, so if you construct a window from an image, then
// you must ensure that the original object outlives the window object.
class Image {
public:
	enum ContructMode {
		ConstructCopy,          // Allocate new buffer that is owned by Image, and copy data into it
		ConstructWindow,        // Assign the data, but do not take ownership of it (ie do not free it)
		ConstructTakeOwnership, // Assume that we now own the given buffer, and will take care of freeing it
	};

	uint8_t*    Data    = nullptr;
	int         Stride  = 0;
	int         Width   = 0;
	int         Height  = 0;
	ImageFormat Format  = ImageFormat::Null;
	bool        OwnData = true;

	Image() {}
	Image(ImageFormat format, ContructMode mode, int stride, void* data, int width, int height);
	Image(ImageFormat format, int width, int height, int stride = 0);
	Image(const Image& b);
	Image(Image&& b);
	~Image();

	Image& operator=(const Image& b);
	Image& operator=(Image&& b);

	void  Reset();                                                          // Free memory, and reset all fields
	void  Alloc(ImageFormat format, int width, int height, int stride = 0); // Allocate memory and initialize data structure
	Image Window(int x, int y, int width, int height) const;                // Returns a window into Image, at the specified rectangle. Does not copy memory. Parent must outlive window.
	void  Fill(uint32_t color);
	void  Fill(Rect32 rect, uint32_t color);
	Image AsType(ImageFormat fmt) const;
	Image HalfSizeCheap() const;             // Downscale by 1/2, in gamma/sRGB space (this is why it's labeled cheap. correct downscale is in linear space, not sRGB)
	void  BoxBlur(int size, int iterations); // Box blur of size [1 + 2 * size], repeated 'iterations' times

	Error SavePng(const std::string& filename, bool withAlpha = true, int zlibLevel = 5) const;
	Error SaveJpeg(const std::string& filename, int quality = 90) const;
	Error SaveFile(const std::string& filename) const;

	uint8_t*        At(int x, int y) { return Data + (y * Stride) + x * BytesPerPixel(); }
	const uint8_t*  At(int x, int y) const { return Data + (y * Stride) + x * BytesPerPixel(); }
	uint32_t*       At32(int x, int y) { return (uint32_t*) (Data + (y * Stride) + x * BytesPerPixel()); }
	const uint32_t* At32(int x, int y) const { return (const uint32_t*) (Data + (y * Stride) + x * BytesPerPixel()); }
	uint8_t*        Line(int y) { return Data + (y * Stride); }
	const uint8_t*  Line(int y) const { return Data + (y * Stride); }
	uint32_t*       Line32(int y) { return (uint32_t*) (Data + (y * Stride)); }
	const uint32_t* Line32(int y) const { return (const uint32_t*) (Data + (y * Stride)); }
	int             BytesPerPixel() const { return gfx::BytesPerPixel(Format); }
	size_t          BytesPerLine() const { return gfx::BytesPerPixel(Format) * Width; }
};

} // namespace gfx
} // namespace imqs