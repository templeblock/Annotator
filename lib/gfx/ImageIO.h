#pragma once

#include "Image.h"

typedef void* tjhandle;

namespace imqs {
namespace gfx {

// ImageIO is cheap to instantiate, but it does cache some state, like libjpegturbo "decompressor",
// so do not share an ImageIO instance between threads.
class ImageIO {
public:
	tjhandle JpegDecomp  = nullptr;
	tjhandle JpegEncoder = nullptr;

	ImageIO();
	~ImageIO();

	// Save to an image format.
	// lossy Q is applicable to jpeg
	// lossless Q is applicable to png
	Error Save(int width, int height, int stride, const void* buf, ImageType type, bool withAlpha, int lossyQ_0_to_100, int losslessQ_1_to_9, void*& encBuf, size_t& encSize);

	// Decodes a png or jpeg image into an RGBA memory buffer
	Error Load(const void* encBuf, size_t encLen, int& width, int& height, void*& buf);

	// Free an encoded buffer. The jpeg-turbo compressor uses a special allocator, so we need to free it specially too.
	static void FreeEncodedBuffer(ImageType type, void* encBuf);

	// Decodes a png image into an RGBA memory buffer
	Error LoadPng(const void* pngBuf, size_t pngLen, int& width, int& height, void*& buf);

	// Encode png
	Error SavePng(bool withAlpha, int width, int height, int stride, const void* buf, int zlibLevel, void*& encBuf, size_t& encSize);

	// Save png to file
	Error SavePngFile(const std::string& filename, bool withAlpha, int width, int height, int stride, const void* buf, int zlibLevel);

	// Decodes a jpeg image into a memory buffer of the desired type. Stride is natural, rounded up to the nearest 4 bytes.
	Error LoadJpeg(const void* jpegBuf, size_t jpegLen, int& width, int& height, void*& buf, TJPF format = TJPF_RGBA);

	// Encode an RGBA buffer to jpeg
	Error SaveJpeg(int width, int height, int stride, const void* buf, int quality_0_to_100, void*& jpegBuf, size_t& jpegSize);
};

} // namespace gfx
} // namespace imqs