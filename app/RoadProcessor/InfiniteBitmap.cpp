#include "pch.h"
#include "InfiniteBitmap.h"
#include "Storage/GCSStorage.h"
#include "Storage/LocalFileStorage.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

Error InfiniteBitmap::Initialize(std::string storageSpec) {
	if (storageSpec.find("gcs://") == 0) {
		auto parts = strings::Split(storageSpec.substr(6), ':');
		if (parts.size() != 2)
			return Error::Fmt("Invalid storage spec. Must be gcs://bucketName:apiKey");
		auto storage = make_shared<GCSStorage>();
		auto err     = storage->Initialize(parts[0], parts[1]);
		if (!err.OK())
			return err;
		Initialize(storage);
		return Error();
	} else {
		auto storage = make_shared<LocalFileStorage>();
		auto err     = storage->Initialize(storageSpec);
		if (!err.OK())
			return err;
		Initialize(storage);
		return Error();
	}
}

void InfiniteBitmap::Initialize(std::shared_ptr<IFileStorage> rawStorage) {
	RawStorage = rawStorage;
}

Error InfiniteBitmap::Load(int zoomLevel, gfx::Rect64 rect, gfx::Image& img, bool* sparseLoadMatrix) const {
	IMQS_ASSERT(rect.x1 % TileSize == 0);
	IMQS_ASSERT(rect.y1 % TileSize == 0);
	IMQS_ASSERT(rect.x2 % TileSize == 0);
	IMQS_ASSERT(rect.y2 % TileSize == 0);
	img.Alloc(ImageFormat::RGBAP, rect.Width(), rect.Height());
	Error    err;
	size_t   rawStripSize = StripSize * TileSize * 4;
	size_t   encBufSize   = StripsPerTile * (sizeof(uint32_t) + LZ4_compressBound(TileSize * StripSize * 4)) + 1; // +1 so we can detect spurious conditions, see comment below
	uint8_t* encBuf       = (uint8_t*) malloc(encBufSize);
	uint8_t* decBuf       = (uint8_t*) malloc(rawStripSize);
	for (int64_t x = rect.x1; x < rect.x2 && err.OK(); x += TileSize) {
		for (int64_t y = rect.y1; y < rect.y2 && err.OK(); y += TileSize) {
			if (sparseLoadMatrix) {
				int tx = int((x - rect.x1) / TileSize);
				int ty = int((y - rect.y1) / TileSize);
				int tw = int((rect.x2 - rect.x1) / TileSize);
				if (!sparseLoadMatrix[ty * tw + tx])
					continue;
			}
			//os::File f;
			//err = f.Open(PathOfTile(x / TileSize, y / TileSize));
			io::Reader* reader = nullptr;
			err                = RawStorage->Open(PathOfTile(zoomLevel, x / TileSize, y / TileSize), reader);
			if (os::IsNotExist(err)) {
				err = Error();
				//img.Fill(Rect32(x - rect.x1, y - rect.y1, x - rect.x1 + TileSize, y - rect.y1 + TileSize), 0);
				continue;
			} else if (!err.OK()) {
				break;
			}
			size_t nRead = encBufSize;
			err          = reader->Read(encBuf, nRead);
			if (!err.OK())
				break;
			if (nRead == encBufSize) {
				// we make our buffer 1 larger than it needs to be, so that we can detect this situation
				err = Error::Fmt("Tile read filed. Read %v bytes, but expected max size of %v", nRead, encBufSize - 1);
				break;
			}
			// first portion is strip sizes (specifically, size of compressed strip. we know the size of the uncompressed strip, because it's constant)
			uint32_t* strips = (uint32_t*) encBuf;
			// after the strip sizes, comes the compressed strips, tightly packed together
			uint8_t* enc = encBuf + StripsPerTile * sizeof(uint32_t);
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

// Note: When this function only supported plain old files on the local filesystem, then it was
// more efficient than it is right now, because it would stream the strips out to the file,
// as it compressed them. However, in order to support GCS, we need to just batch up all writes
// into a single API call. I never measured the performance loss due to this change, but I
// suspect it's negligible.
Error InfiniteBitmap::Save(int zoomLevel, gfx::Rect64 rect, const gfx::Image& img, bool* sparseSaveMatrix) const {
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
	uint32_t* strips       = new uint32_t[StripsPerTile];
	//ohash::map<string, string> headers      = {
	//    {"Content-Type", "road-tile-1"},
	//};
	io::Buffer writeBuf;
	for (int64_t x = rect.x1; x < rect.x2 && err.OK(); x += TileSize) {
		for (int64_t y = rect.y1; y < rect.y2 && err.OK(); y += TileSize) {
			if (sparseSaveMatrix) {
				int tx = int((x - rect.x1) / TileSize);
				int ty = int((y - rect.y1) / TileSize);
				int tw = int((rect.x2 - rect.x1) / TileSize);
				if (!sparseSaveMatrix[ty * tw + tx])
					continue;
			}
			//os::File f;
			//err = f.Create(PathOfTile(x / TileSize, y / TileSize));
			//if (!err.OK())
			//	break;
			writeBuf.Len = 0;
			// this is just a seek (we come back at the end and write the actual values)
			err = writeBuf.Write(strips, sizeof(uint32_t) * StripsPerTile);
			if (!err.OK())
				break;
			for (int strip = 0; strip < StripsPerTile; strip++) {
				// copy raw bytes into contiguous buffer, so that we can send it to the compressor
				for (int i = 0; i < StripSize; i++)
					memcpy(decBuf + i * TileSize * 4, img.At(x - rect.x1, y - rect.y1 + strip * StripSize + i), TileSize * 4);
				int r = LZ4_compress_default((const char*) decBuf, (char*) encBuf, StripSize * TileSize * 4, encBufSize);
				IMQS_ASSERT(r > 0);
				//IMQS_ASSERT(r > 0 && r <= 65535);
				strips[strip] = (uint32_t) r;
				err           = writeBuf.Write(encBuf, r);
				if (!err.OK())
					break;
			}
			// write the final strip size values
			//err = f.Seek(0, io::SeekWhence::Begin);
			//if (!err.OK())
			//	break;
			//err = f.Write(strips, sizeof(uint32_t) * StripsPerTile);
			//if (!err.OK())
			//	break;
			memcpy(writeBuf.Buf, strips, sizeof(uint32_t) * StripsPerTile);
			err = RawStorage->Create(PathOfTile(zoomLevel, x / TileSize, y / TileSize), FileStorageClass::Regional, writeBuf.Buf, writeBuf.Len);
			if (!err.OK())
				break;
		}
	}
	delete[] strips;
	free(encBuf);
	free(decBuf);
	return err;
}

Error InfiniteBitmap::CreateWebTiles(int zoomLevel) {
	string                   RootDir = "foobar"; // HACK to get this compiling during refactor
	vector<os::FindFileItem> all;
	auto                     err = os::FindFiles(RootDir, [&all](const os::FindFileItem& item) -> bool {
        if (item.IsDir)
            return false;
        if (item.Name.find("t-") == 0 && item.Name.find(".lz4") != -1)
            all.push_back(item);
        return true;
    });
	if (!err.OK())
		return err;

	int webTileSize = 256;

	// first zoom level
	//int firstLevel = 25;

	int          jpegQuality  = 90;
	JpegSampling jpegSampling = JpegSampling::Samp422;

	// at level 10 this puts us somewhere in east africa
	//int64_t xOffset = 18000;
	//int64_t yOffset = 19000;
	int64_t xOffset = 0;
	int64_t yOffset = 0;

	tsf::print("Processing...\n");

	gfx::Image img;
	auto       outDir = path::Join(RootDir, "webtiles");
	for (size_t i = 0; i < all.size(); i++) {
		const auto& t = all[i];
		// typical name: t--0000002--0000144.lz4
		int64_t x = AtoI64(t.Name.substr(2, 8).c_str());
		int64_t y = AtoI64(t.Name.substr(11, 8).c_str());

		mutex loadLock;
		bool  isLoaded     = false;
		auto  ensureLoaded = [&]() -> Error {
            lock_guard<mutex> lock(loadLock);
            if (!isLoaded) {
                auto err = Load(zoomLevel, Rect64(x * TileSize, y * TileSize, (x + 1) * TileSize, (y + 1) * TileSize), img);
                if (!err.OK())
                    return err;
            }
            return Error();
		};

		int64_t absMacroX = xOffset + x * (TileSize / webTileSize);
		int64_t absMacroY = yOffset + y * (TileSize / webTileSize);

		// native resolution
		int nchunk = TileSize / webTileSize;

		vector<Point32> chunks;
		for (int cy = 0; cy < nchunk; cy++) {
			for (int cx = 0; cx < nchunk; cx++) {
				chunks.emplace_back(cx, cy);
			}
		}

		mutex errorLock;
		auto  setLockedError = [&](Error newErr) {
            lock_guard<mutex> lock(errorLock);
            err = newErr;
		};

		int chunks_size = chunks.size();
#pragma omp parallel for
		for (int i = 0; i < chunks_size; i++) {
			{
				// this is our equivalent of a 'break' statement, because we're not allowed to break out of an OMP loop
				lock_guard<mutex> lock(errorLock);
				if (!err.OK())
					continue;
			}
			int     cx         = chunks[i].x;
			int     cy         = chunks[i].y;
			int64_t absX       = absMacroX + cx;
			int64_t absY       = absMacroY + cy;
			string  outTileDir = path::Join(outDir, ItoA(zoomLevel), ItoA(absX));
			auto    e          = os::MkDirAll(outTileDir);
			if (!e.OK()) {
				setLockedError(e);
				continue;
			}
			string             outTileFile = path::Join(outTileDir, tsf::fmt("%d.jpeg", absY));
			os::FileAttributes attribs;
			e = os::Stat(outTileFile, attribs);
			if (e.OK() && attribs.TimeModify > t.TimeModify) {
				// tile already exists, and is newer than the lz4 tile
				continue;
			}
			e = ensureLoaded();
			if (!e.OK()) {
				setLockedError(e);
				continue;
			}
			auto chunk = img.Window(cx * webTileSize, cy * webTileSize, webTileSize, webTileSize);
			e          = chunk.SaveJpeg(outTileFile, jpegQuality, jpegSampling);
			if (!e.OK()) {
				setLockedError(e);
				continue;
			}
		}
		if (!err.OK())
			return err;
		tsf::print("\rFinished %d/%d", i, all.size());
		fflush(stdout);
	}

	return Error();
}

std::string InfiniteBitmap::PathOfTile(int zoomLevel, int64_t tx, int64_t ty) const {
	return tsf::fmt("t-%02d-%08d-%08d.lz4", zoomLevel, tx, ty);
}

int64_t InfiniteBitmap::RoundDown64(int64_t x, int32_t y) {
	// C rounds to zero, even for negative numbers.
	// Here, we round towards negative infinity.
	if (x < 0)
		return ((x - y + 1) / y) * y;
	else
		return (x / y) * y;
}

int64_t InfiniteBitmap::RoundUp64(int64_t x, int32_t y) {
	// Because of C's rounding towards zero, our job becomes simple when rounding up
	if (x < 0)
		return (x / y) * y;
	else
		return ((x + y - 1) / y) * y;
}

} // namespace roadproc
} // namespace imqs
