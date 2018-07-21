#include "pch.h"
#include "Image.h"
#include "ImageIO.h"

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
	if (this != &b) {
		Reset();
		Width   = b.Width;
		Height  = b.Height;
		Stride  = b.Stride;
		Data    = (uint8_t*) imqs_malloc_or_die(Height * Stride);
		Format  = b.Format;
		OwnData = true;
		for (int y = 0; y < Height; y++)
			memcpy(Line(y), b.Line(y), BytesPerLine());
	}
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
	if (stride == 0) {
		stride = gfx::BytesPerPixel(format) * width;
		stride = 4 * ((stride + 3) / 4); // round up to multiple of 4
	}

	IMQS_ASSERT(stride >= gfx::BytesPerPixel(format) * width);

	if (Width == width && Height == height && Stride == stride && OwnData) {
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

Image Image::Window(Rect32 rect) const {
	return Window(rect.x1, rect.y1, rect.Width(), rect.Height());
}

void Image::Fill(uint32_t color) {
	Fill(Rect32(0, 0, Width, Height), color);
}

void Image::Fill(Rect32 rect, uint32_t color) {
	rect.x1 = math::Clamp(rect.x1, 0, Width);
	rect.y1 = math::Clamp(rect.y1, 0, Height);
	rect.x2 = math::Clamp(rect.x2, 0, Width);
	rect.y2 = math::Clamp(rect.y2, 0, Height);
	for (int y = rect.y1; y < rect.y2; y++) {
		uint32_t* dst = At32(rect.x1, y);
		size_t    x2  = rect.x2;
		for (size_t x = rect.x1; x < x2; x++)
			dst[x] = color;
	}
}

#define MAKE_IMAGE_FORMAT_PAIR(a, b) (((uint32_t) a << 16) | (uint32_t) b)

Image Image::AsType(ImageFormat fmt) const {
	Image r;
	r.Alloc(fmt, Width, Height);
	uint32_t combo = MAKE_IMAGE_FORMAT_PAIR(Format, fmt);
	switch (combo) {
	case MAKE_IMAGE_FORMAT_PAIR(ImageFormat::RGBAP, ImageFormat::Gray):
	case MAKE_IMAGE_FORMAT_PAIR(ImageFormat::RGBA, ImageFormat::Gray):
		for (int y = 0; y < Height; y++) {
			size_t         w   = Width;
			const uint8_t* src = Line(y);
			uint8_t*       dst = r.Line(y);
			for (size_t x = 0; x < w; x++) {
				dst[0] = Color8(src[0], src[1], src[2], src[3]).Lum();
				dst += 1;
				src += 4;
			}
		}
		break;
	case MAKE_IMAGE_FORMAT_PAIR(ImageFormat::Gray, ImageFormat::RGBA):
	case MAKE_IMAGE_FORMAT_PAIR(ImageFormat::Gray, ImageFormat::RGBAP):
		for (int y = 0; y < Height; y++) {
			size_t         w   = Width;
			const uint8_t* src = Line(y);
			uint8_t*       dst = r.Line(y);
			for (size_t x = 0; x < w; x++) {
				dst[0] = src[0];
				dst[1] = src[0];
				dst[2] = src[0];
				dst[3] = 255;
				dst += 4;
				src += 1;
			}
		}
		break;
	default:
		IMQS_DIE();
	}
	return r;
}

Image Image::HalfSizeCheap() const {
	Image half;
	half.Alloc(Format, Width / 2, Height / 2);
	for (int y = 0; y < half.Height; y++) {
		auto   srcA = Line(y * 2);     // top line
		auto   srcB = Line(y * 2 + 1); // bottom line
		auto   dstP = half.Line(y);
		size_t dstW = half.Width;
		if (NumChannels() == 4) {
			for (size_t x = 0; x < dstW; x++) {
				uint32_t r = ((uint32_t) srcA[0] + (uint32_t) srcA[4] + (uint32_t) srcB[0] + (uint32_t) srcB[4]) >> 2;
				uint32_t g = ((uint32_t) srcA[1] + (uint32_t) srcA[5] + (uint32_t) srcB[1] + (uint32_t) srcB[5]) >> 2;
				uint32_t b = ((uint32_t) srcA[2] + (uint32_t) srcA[6] + (uint32_t) srcB[2] + (uint32_t) srcB[6]) >> 2;
				uint32_t a = ((uint32_t) srcA[3] + (uint32_t) srcA[7] + (uint32_t) srcB[3] + (uint32_t) srcB[7]) >> 2;
				dstP[0]    = r;
				dstP[1]    = g;
				dstP[2]    = b;
				dstP[3]    = a;
				srcA += 8;
				srcB += 8;
				dstP += 4;
			}
		} else if (NumChannels() == 1) {
			for (size_t x = 0; x < dstW; x++) {
				dstP[0] = ((uint32_t) srcA[0] + (uint32_t) srcA[1] + (uint32_t) srcB[0] + (uint32_t) srcB[1]) >> 2;
				srcA += 2;
				srcB += 2;
				dstP += 1;
			}
		} else {
			IMQS_DIE();
		}
	}
	return half;
}

// This is crazy slow. It needs to be vectorized.
Image Image::HalfSizeLinear() const {
	Image half;
	half.Alloc(Format, Width / 2, Height / 2);
	for (int y = 0; y < half.Height; y++) {
		auto   srcA = Line(y * 2);     // top line
		auto   srcB = Line(y * 2 + 1); // bottom line
		auto   dstP = half.Line(y);
		size_t dstW = half.Width;
		if (NumChannels() == 4) {
			for (size_t x = 0; x < dstW; x++) {
				float    r = 0.25f * (Color8::SRGBtoLinearU8(srcA[0]) + Color8::SRGBtoLinearU8(srcA[4]) + Color8::SRGBtoLinearU8(srcB[0]) + Color8::SRGBtoLinearU8(srcB[4]));
				float    g = 0.25f * (Color8::SRGBtoLinearU8(srcA[1]) + Color8::SRGBtoLinearU8(srcA[5]) + Color8::SRGBtoLinearU8(srcB[1]) + Color8::SRGBtoLinearU8(srcB[5]));
				float    b = 0.25f * (Color8::SRGBtoLinearU8(srcA[2]) + Color8::SRGBtoLinearU8(srcA[6]) + Color8::SRGBtoLinearU8(srcB[2]) + Color8::SRGBtoLinearU8(srcB[6]));
				uint32_t a = ((uint32_t) srcA[3] + (uint32_t) srcA[7] + (uint32_t) srcB[3] + (uint32_t) srcB[7]) >> 2;
				dstP[0]    = Color8::LinearToSRGBU8(r);
				dstP[1]    = Color8::LinearToSRGBU8(g);
				dstP[2]    = Color8::LinearToSRGBU8(b);
				dstP[3]    = a;
				srcA += 8;
				srcB += 8;
				dstP += 4;
			}
		} else if (NumChannels() == 1) {
			for (size_t x = 0; x < dstW; x++) {
				float a = Color8::SRGBtoLinearU8(srcA[0]);
				float b = Color8::SRGBtoLinearU8(srcA[1]);
				float c = Color8::SRGBtoLinearU8(srcB[0]);
				float d = Color8::SRGBtoLinearU8(srcB[1]);
				a       = 0.25f * (a + b + c + d);
				dstP[0] = Color8::LinearToSRGBU8(a);
				srcA += 2;
				srcB += 2;
				dstP += 1;
			}
		} else {
			IMQS_DIE();
		}
	}
	return half;
}

struct Color16 {
	union {
		struct
		{
#if ENDIANLITTLE
			uint16_t a, b, g, r;
#else
			uint16_t r : 16;
			uint16_t g : 16;
			uint16_t b : 16;
			uint16_t a : 16;
#endif
		};
		uint64_t u;
	};
	Color16() {}
	Color16(Color8 x) : r(x.r), g(x.g), b(x.b), a(x.a) {}
	Color16(uint16_t r, uint16_t g, uint16_t b, uint16_t a) : r(r), g(g), b(b), a(a) {}

	Color8 ToColor8() const { return Color8(r, g, b, a); }
	operator Color8() const { return Color8(r, g, b, a); }
};

inline Color16 operator/(Color16 x, uint16_t div) {
	return Color16(x.r / div, x.g / div, x.b / div, x.a / div);
}

inline Color16 operator+(Color16 x, Color16 y) {
	return Color16(x.r + y.r, x.g + y.g, x.b + y.b, x.a + y.a);
}

inline Color16 operator-(Color16 x, Color16 y) {
	return Color16(x.r - y.r, x.g - y.g, x.b - y.b, x.a - y.a);
}

static void BoxBlurGray(uint8_t* src, uint8_t* dst, int width, int blurSize, int iterations) {
	for (int iter = 0; iter < iterations; iter++) {
		unsigned sum = 0;
		if (blurSize == 1) {
			dst[0] = ((unsigned) src[0] + (unsigned) src[1]) >> 1;
			dst[1] = ((unsigned) src[0] + (unsigned) src[1] + (unsigned) src[2]) / 3;
			sum    = (unsigned) src[0] + (unsigned) src[1] + (unsigned) src[2];
		}
		size_t x = blurSize + 1;
		size_t w = width - blurSize;
		for (; x < w; x++) {
			sum    = sum - src[x - 2] + src[x + 1];
			dst[x] = sum / 3;
		}
		if (blurSize == 1) {
			sum    = sum - src[x - 2] + src[w - 1];
			dst[x] = sum / 3;
		}
		std::swap(src, dst);
	}
}

static void BoxBlurRGBA(Color8* src, Color8* dst, int width, int blurSize, int iterations) {
	for (int iter = 0; iter < iterations; iter++) {
		Color16 sum(0, 0, 0, 0);
		if (blurSize == 1) {
			dst[0] = (Color16(src[0]) + Color16(src[1])) / 2;
			dst[1] = (Color16(src[0]) + Color16(src[1]) + Color16(src[2])) / 3;
			sum    = Color16(src[0]) + Color16(src[1]) + Color16(src[2]);
		}
		size_t x = blurSize + 1;
		size_t w = width - blurSize;
		for (; x < w; x++) {
			sum = sum - Color16(src[x - 2]) + Color16(src[x + 1]);
			//sum.r = sum.r - src[x - 2].r + src[x + 1].r;
			//sum.g = sum.g - src[x - 2].g + src[x + 1].g;
			//sum.b = sum.b - src[x - 2].b + src[x + 1].b;
			//sum.a = sum.a - src[x - 2].a + src[x + 1].a;
			dst[x] = sum / 3;
		}
		if (blurSize == 1) {
			sum    = sum - src[x - 2] + src[w - 1];
			dst[x] = sum / 3;
		}
		std::swap(src, dst);
	}
}

void Image::BoxBlur(int size, int iterations) {
	IMQS_ASSERT(NumChannels() == 1 || NumChannels() == 4);
	IMQS_ASSERT(Width >= size * 2 + 1);
	IMQS_ASSERT(Height >= size * 2 + 1);
	IMQS_ASSERT(size == 1); // just haven't bothered to code up other sizes, and it's only a hassle because of the left/right edge cases
	bool isRGBA = NumChannels() == 4;

	uint8_t* buf1 = (uint8_t*) imqs_malloc_or_die((max(Width, Height) + 1) * NumChannels());
	uint8_t* buf2 = (uint8_t*) imqs_malloc_or_die((Height + 1) * NumChannels());

	// these are just sentinels to make sure we're not overwriting memory
	buf1[max(Width, Height) * NumChannels()] = 254;
	buf2[Height * NumChannels()]             = 254;

	bool evenIterations = (unsigned) iterations % 2 == 0;

	for (int y = 0; y < Height; y++) {
		uint8_t* src = Line(y);
		if (isRGBA)
			BoxBlurRGBA((Color8*) src, (Color8*) buf1, Width, size, iterations);
		else
			BoxBlurGray(src, buf1, Width, size, iterations);
		if (!evenIterations)
			memcpy(src, buf1, Width * NumChannels());
	}

	for (int x = 0; x < Width; x++) {
		// For the verticals, first copy each line into a buffer, so that we can run multiple iterations fast
		size_t h      = Height;
		int    stride = Stride;
		if (isRGBA) {
			uint32_t* src = At32(x, 0);
			uint32_t* buf = (uint32_t*) buf1;
			for (size_t y = 0; y < h; y++, (char*&) src += Stride)
				buf[y] = *src;
		} else {
			uint8_t* src = At(x, 0);
			for (size_t y = 0; y < h; y++, src += Stride)
				buf1[y] = *src;
		}

		if (isRGBA)
			BoxBlurRGBA((Color8*) buf1, (Color8*) buf2, Height, size, iterations);
		else
			BoxBlurGray(buf1, buf2, Height, size, iterations);

		// copy back out
		if (isRGBA) {
			uint32_t* src   = At32(x, 0);
			uint32_t* final = evenIterations ? (uint32_t*) buf1 : (uint32_t*) buf2;
			for (size_t y = 0; y < h; y++, (char*&) src += Stride)
				*src = final[y];
		} else {
			uint8_t* src   = At(x, 0);
			uint8_t* final = evenIterations ? buf1 : buf2;
			for (size_t y = 0; y < h; y++, src += Stride)
				*src = final[y];
		}
	}

	IMQS_ASSERT(buf1[max(Width, Height) * NumChannels()] == 254); // sentils to make sure we haven't overwritten memory
	IMQS_ASSERT(buf2[Height * NumChannels()] == 254);

	free(buf1);
	free(buf2);
}

void Image::CopyFrom(const Image& src, Rect32 srcRect, Rect32 dstRect) {
	IMQS_ASSERT(srcRect.Width() == dstRect.Width());
	IMQS_ASSERT(srcRect.Height() == dstRect.Height());
	IMQS_ASSERT(src.Format == Format);

	if (srcRect.x1 < 0) {
		dstRect.x1 -= srcRect.x1;
		srcRect.x1 = 0;
	}
	if (srcRect.y1 < 0) {
		dstRect.y1 -= srcRect.y1;
		srcRect.y1 = 0;
	}
	if (srcRect.x2 > src.Width) {
		dstRect.x2 -= srcRect.x2 - src.Width;
		srcRect.x2 = src.Width;
	}
	if (srcRect.y2 > src.Height) {
		dstRect.y2 -= srcRect.x2 - src.Height;
		srcRect.y2 = src.Height;
	}

	if (dstRect.x1 < 0) {
		srcRect.x1 -= dstRect.x1;
		dstRect.x1 = 0;
	}
	if (dstRect.y1 < 0) {
		srcRect.y1 -= dstRect.y1;
		dstRect.y1 = 0;
	}
	if (dstRect.x2 > Width) {
		srcRect.x2 -= dstRect.x2 - Width;
		dstRect.x2 = Width;
	}
	if (dstRect.y2 > Height) {
		srcRect.y2 -= dstRect.x2 - Height;
		dstRect.y2 = Height;
	}

	if (srcRect.Width() < 0 || srcRect.Height() < 0)
		return;

	for (int y = 0; y < srcRect.Height(); y++)
		memcpy(At(dstRect.x1, dstRect.y1 + y), src.At(srcRect.x1, srcRect.y1 + y), srcRect.Width() * BytesPerPixel());
}

void Image::CopyFrom(const Image& src, Rect32 srcRect, int dstX, int dstY) {
	CopyFrom(src, srcRect, Rect32(dstX, dstY, dstX + srcRect.Width(), dstY + srcRect.Height()));
}

Error Image::SavePng(const std::string& filename, bool withAlpha, int zlibLevel) const {
	if (Format == ImageFormat::Gray) {
		auto copy = AsType(ImageFormat::RGBA);
		return copy.SavePng(filename, false, zlibLevel);
	}
	return ImageIO::SavePngFile(filename, withAlpha, Width, Height, Stride, Data, zlibLevel);
}

Error Image::SaveJpeg(const std::string& filename, int quality) const {
	if (Format == ImageFormat::Gray) {
		auto copy = AsType(ImageFormat::RGBA);
		return copy.SaveJpeg(filename, quality);
	}
	return ImageIO::SaveJpegFile(filename, Width, Height, Stride, Data, quality);
}

Error Image::SaveFile(const std::string& filename) const {
	auto ext = strings::tolower(path::Extension(filename));
	if (ext == ".png")
		return SavePng(filename, true, 1);
	else if (ext == ".jpeg" || ext == ".jpg")
		return SaveJpeg(filename);
	else
		return Error::Fmt("Unknown image file format '%v'", ext);
}

} // namespace gfx
} // namespace imqs
