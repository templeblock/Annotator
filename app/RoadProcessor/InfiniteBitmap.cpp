#include "pch.h"
#include "InfiniteBitmap.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

static int64_t RoundDown64(int64_t x, int32_t y) {
	// C rounds to zero, even for negative numbers.
	// Here, we round towards negative infinity.
	if (x < 0)
		return ((x - y + 1) / y) * y;
	else
		return (x / y) * y;
}

Error InfiniteBitmap::Initialize(std::string rootDir) {
	RootDir = rootDir;
	return os::MkDirAll(rootDir);
}

Error InfiniteBitmap::Load(gfx::Rect64 rect, gfx::Image& img) const {
	IMQS_ASSERT(rect.x1 % TileSize == 0);
	IMQS_ASSERT(rect.y1 % TileSize == 0);
	IMQS_ASSERT(rect.x2 % TileSize == 0);
	IMQS_ASSERT(rect.y2 % TileSize == 0);
	img.Alloc(ImageFormat::RGBA, rect.Width(), rect.Height());
	Error    err;
	size_t   rawStripSize = StripSize * TileSize * 4;
	size_t   encBufSize   = StripsPerTile * sizeof(uint16_t) + LZ4_compressBound(TileSize * TileSize * 4) + 1; // +1 so we can detect spurious conditions, see comment below
	uint8_t* encBuf       = (uint8_t*) malloc(encBufSize);
	uint8_t* decBuf       = (uint8_t*) malloc(rawStripSize);
	for (int64_t x = RoundDown64(rect.x1, TileSize); x < rect.x2 && err.OK(); x += TileSize) {
		for (int64_t y = RoundDown64(rect.y1, TileSize); y < rect.y2 && err.OK(); y += TileSize) {
			os::File f;
			err = f.Open(PathOfTile(x / TileSize, y / TileSize));
			if (os::IsNotExist(err)) {
				err = Error();
				img.Fill(0);
				continue;
			} else if (!err.OK()) {
				break;
			}
			size_t nRead = encBufSize;
			err          = f.Read(encBuf, nRead);
			if (!err.OK())
				break;
			if (nRead == encBufSize) {
				// we make our buffer 1 larger than it needs to be, so that we can detect this situation
				err = Error::Fmt("Tile read filed. Read %v bytes, but expected max size of %v", nRead, encBufSize - 1);
				break;
			}
			// first portion is strip sizes (specifically, size of compressed strip. we know the size of the uncompressed strip, because it's constant)
			uint16_t* strips = (uint16_t*) encBuf;
			// after the strip sizes, comes the compressed strips, tightly packed together
			uint8_t* enc = encBuf + StripsPerTile * sizeof(uint16_t);
			for (int strip = 0; strip < StripsPerTile; strip++) {
				int r = LZ4_decompress_fast((const char*) enc, (char*) decBuf, rawStripSize);
				if (r != strips[strip]) {
					err = Error::Fmt("Failed to decompress tile %v,%v, strip %v: LZ4 error code %v", x / TileSize, y / TileSize, strip, r);
					break;
				}
				for (int i = 0; i < StripSize; i++)
					memcpy(img.At(x - rect.x1, y - rect.y1 + strip * StripSize + i), decBuf + i * TileSize * 4, TileSize * 4);
				enc += strips[strip];
			}
		}
	}
	free(encBuf);
	free(decBuf);
	return err;
}

Error InfiniteBitmap::Save(gfx::Rect64 rect, const gfx::Image& img) const {
	IMQS_ASSERT(rect.x1 % TileSize == 0);
	IMQS_ASSERT(rect.y1 % TileSize == 0);
	IMQS_ASSERT(rect.x2 % TileSize == 0);
	IMQS_ASSERT(rect.y2 % TileSize == 0);
	IMQS_ASSERT(img.Width >= rect.Width());
	IMQS_ASSERT(img.Height >= rect.Height());
	Error     err;
	size_t    rawStripSize = StripSize * TileSize * 4;
	size_t    encBufSize   = LZ4_compressBound(TileSize * StripSize * 4) + 1;
	uint8_t*  encBuf       = (uint8_t*) malloc(encBufSize);
	uint8_t*  decBuf       = (uint8_t*) malloc(rawStripSize);
	uint16_t* strips       = new uint16_t[StripsPerTile];
	for (int64_t x = RoundDown64(rect.x1, TileSize); x < rect.x2 && err.OK(); x += TileSize) {
		for (int64_t y = RoundDown64(rect.y1, TileSize); y < rect.y2 && err.OK(); y += TileSize) {
			os::File f;
			err = f.Create(PathOfTile(x / TileSize, y / TileSize));
			if (!err.OK())
				break;
			// this is just a seek
			err = f.Write(strips, sizeof(uint16_t) * StripsPerTile);
			if (!err.OK())
				break;
			for (int strip = 0; strip < StripsPerTile; strip++) {
				// copy raw bytes into contiguous buffer, so that we can send it to the compressor
				for (int i = 0; i < StripSize; i++)
					memcpy(decBuf + i * TileSize * 4, img.At(x - rect.x1, y - rect.y1 + strip * StripSize + i), TileSize * 4);
				int r = LZ4_compress_default((const char*) decBuf, (char*) encBuf, StripSize * TileSize * 4, encBufSize);
				IMQS_ASSERT(r > 0 && r <= 65535);
				strips[strip] = (uint16_t) r;
				err           = f.Write(encBuf, r);
				if (!err.OK())
					break;
			}
			// write the final strip size values
			err = f.Seek(0, io::SeekWhence::Begin);
			if (!err.OK())
				break;
			err = f.Write(strips, sizeof(uint16_t) * StripsPerTile);
			if (!err.OK())
				break;
		}
	}
	delete[] strips;
	free(encBuf);
	free(decBuf);
	return err;
}

std::string InfiniteBitmap::PathOfTile(int64_t tx, int64_t ty) const {
	return path::Join(RootDir, tsf::fmt("t-%08d-%08d.lz4", tx, ty));
}

} // namespace roadproc
} // namespace imqs