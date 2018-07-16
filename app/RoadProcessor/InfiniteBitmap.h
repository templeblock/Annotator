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
in 64 strips. Each strip is 16 pixels high. If we multiply 64
* 16, we get 1024, which is our tile size. An uncompressed
strip is 16 * 1024 * 4 = 64K.

A compressed tile first starts with an array of type uint32[64],
where each entry marks the size of the compressed strip.
By compressing in strips, we're able to stream the decompression,
without going through an extra 4 MB memory buffer, that would
be necessary to store an entire tile of 1024*1024*4.
*/
class InfiniteBitmap {
public:
	/*
	struct Tile {
		int64_t  X    = 0;
		int64_t  Y    = 0;
		uint32_t Hash = 0;
		Tile() {}
		Tile(int64_t x, int64_t y) : X(x), Y(y) {
			int64_t buf[2] = {x, y};
			Hash           = fnv_32a_buf(buf, 2 * sizeof(int64_t));
		}

		bool operator==(const Tile& t) const { return X == t.X && Y == t.Y; }
		bool operator!=(const Tile& t) const { return !(*this == t); }
	};
	*/

	int TileSize = 1024;

	Error Initialize(std::string rootDir);
	Error Load(gfx::Rect64 rect, gfx::Image& img) const;       // rect must be aligned to TileSize
	Error Save(gfx::Rect64 rect, const gfx::Image& img) const; // rect must be aligned to TileSize
	Error CreateWebTiles();

	static int64_t RoundDown64(int64_t x, int32_t y);
	static int64_t RoundUp64(int64_t x, int32_t y);

private:
	std::string RootDir;
	int         StripSize     = 16;
	int         StripsPerTile = TileSize / StripSize;

	Error       DownscaleWebTiles(std::string outDir, int level);
	std::string PathOfTile(int64_t tx, int64_t ty) const;
};

} // namespace roadproc
} // namespace imqs

/*
namespace ohash {
template <>
inline hashkey_t gethashcode(const imqs::roadproc::InfiniteBitmap::Tile& t) {
	return (hashkey_t) t.Hash;
}
} // namespace ohash
*/
