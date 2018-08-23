#pragma once

#include "Storage/FileStorage.h"

namespace imqs {
namespace roadproc {

/*
InfiniteBitmap is an abstraction over an infinitely large bitmap,
that can serve up a portion of that bitmap, and then commit that
portion back to disk. Of course when we say infinite, we do in
fact mean "planet size". This was built to store the rasterized,
stitched imagery coming out of the camera.

The coordinate system is top-down, like a regular bitmap. Tiles
are stored on disk, compressed with LZ4. A tile is compressed
in 64 strips. Each strip is 16 pixels high. If we multiply 64
* 16, we get 1024, which is our tile size. An uncompressed
strip is 16 * 1024 * 4 = 64K.

A compressed tile first starts with an array of type uint32[64],
where each entry marks the size of the compressed strip.
By compressing in strips, we're able to stream the decompression,
without going through an extra 4 MB memory buffer, which would
be necessary to store an entire tile of 1024*1024*4.
*/
class InfiniteBitmap {
public:
	int TileSize = 1024;

	// Initialize with either of these two options:
	// 1. /path/to/local/filesystem
	// 2. gcs://bucketName:apiKey
	Error Initialize(std::string storageSpec);
	void  Initialize(std::shared_ptr<IFileStorage> rawStorage);

	// rect must be aligned to TileSize (ie must be loading exact tile multiples)
	// See Save() for an explanation of sparseLoadMatrix.
	Error Load(int zoomLevel, gfx::Rect64 rect, gfx::Image& img, bool* sparseLoadMatrix = nullptr) const;

	// Save a bitmap containing one or more tiles to storage.
	// rect must be aligned to TileSize (ie must be saving exact tile multiples)
	// If sparseSaveMatrix is not null, then it is a 2D array (row major), containing
	// a bool value for every tile in img. The tile is only written if the value
	// in sparseSaveMatrix is true.
	Error Save(int zoomLevel, gfx::Rect64 rect, const gfx::Image& img, bool* sparseSaveMatrix = nullptr) const;

	Error CreateWebTiles(int zoomLevel);

	static int64_t RoundDown64(int64_t x, int32_t y);
	static int64_t RoundUp64(int64_t x, int32_t y);

private:
	std::shared_ptr<IFileStorage> RawStorage    = nullptr;
	int                           StripSize     = 16;
	int                           StripsPerTile = TileSize / StripSize;

	std::string PathOfTile(int zoomLevel, int64_t tx, int64_t ty) const;
};

} // namespace roadproc
} // namespace imqs
