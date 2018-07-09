#pragma once

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
in 64 strips. Each strip is 16 pixels high. 16 * 64 = 1024.
A compressed tile first starts with an array of type uint16[64],
where each entry marks the size of the compressed strip.
By compressing in strips, we're able to stream the decompression,
without going through an extra 4 MB memory buffer of 1024*1024*4.
*/
class InfiniteBitmap {
public:
	Error Initialize(std::string rootDir);
	Error Load(gfx::Rect64 rect, gfx::Image& img) const;
	Error Save(gfx::Rect64 rect, const gfx::Image& img) const;

private:
	std::string RootDir;
	int         TileSize      = 1024;
	int         StripSize     = 16; // 4 * StripSize must be less than 64k, because our strip sizes are stored as uint16
	int         StripsPerTile = TileSize / StripSize;

	std::string PathOfTile(int64_t tx, int64_t ty) const;
};

} // namespace roadproc
} // namespace imqs