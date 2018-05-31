#include "pch.h"
#include "compress/zlib.h"

namespace imqs {
namespace compress {
namespace zlib {

// On Windows zlib takes 32-bit sizes, so you'll need to use a streaming mode for
// compressions of large buffers. The zlib sizes are unsigned, so by clamping to half
// the unsigned range, we're not in danger of something like a few bytes of overflow,
// due to an incompressible source buffer.
static const size_t MaxZLibDataSize = (size_t) 0x7FFFFFFF;

IMQS_PAL_API Error Compress(const void* raw, size_t rawLen, void*& enc, size_t& encLen, uint32_t flags) {
	IMQS_ASSERT(rawLen < MaxZLibDataSize);

	// gzip header needs a few more bytes than compressBounds returns to us. 20 seems to be enough.
	// As far as I can tell, the gzip header is 10 bytes, and the deflate header is 2 bytes, for the
	// parameters that we use inside DeflateInit(). So we make it 20, to be sure.
	size_t tmpSize = ::compressBound((uLong) rawLen) + 20; // +20 so that we always get Z_STREAM_END and not Z_OK
	void*  tmp     = malloc(tmpSize);
	if (!tmp)
		return Error::Fmt("Out of memory allocating %v bytes for zlib compression", tmpSize);

	z_stream stream;
	auto     err = DeflateInit(stream, flags);
	if (!err.OK())
		return err;

	stream.avail_in = (uInt) rawLen;
	stream.next_in  = (Byte*) raw;

	stream.avail_out = (uInt) tmpSize;
	stream.next_out  = (Byte*) tmp;

	int r = deflate(&stream, Z_FINISH);
	deflateEnd(&stream);
	if (r != Z_STREAM_END) {
		free(tmp);
		return Error::Fmt("Error compressing %v bytes with zlib: %v", rawLen, stream.msg ? stream.msg : "?");
	}

	encLen = tmpSize - stream.avail_out;

	if (!!(flags & FlagShrinkBuffer)) {
		enc = malloc(encLen);
		if (!enc) {
			free(tmp);
			return Error::Fmt("Out of memory allocating %v bytes for zlib compression", encLen);
		}
		memcpy(enc, tmp, encLen);
		free(tmp);
	} else {
		enc = tmp;
	}

	return Error();
}

IMQS_PAL_API Error Uncompress(const void* comp, size_t compLen, void*& raw, size_t& rawLen) {
	if (compLen == 0)
		return Error("Buffer to uncompress (zlib) is empty");

	z_stream in_s;
	memset(&in_s, 0, sizeof(in_s));
	in_s.next_in  = (Bytef*) comp;
	in_s.avail_in = (uInt) compLen;
	inflateInit2(&in_s, 15 + 32); // 15 + 32 => auto detect zlib/gzip

	size_t maxRawLen = rawLen;
	if (raw == nullptr && rawLen == 0)
		maxRawLen = -1;

	bool   selfAlloc = raw == nullptr;
	size_t tryLen    = selfAlloc ? std::min(maxRawLen, compLen * 2) : rawLen;

	char* out = (char*) raw;
	while (true) {
		if (selfAlloc) {
			char* newOut = (char*) realloc(out, tryLen);
			if (!newOut) {
				if (selfAlloc)
					free(out);
				return Error::Fmt("Out of memory decompressing zlib stream");
			}
			out = newOut;
		}
		in_s.next_out  = (Bytef*) (out + in_s.total_out);
		in_s.avail_out = (uInt) tryLen;
		int r          = inflate(&in_s, Z_SYNC_FLUSH);
		if (r == Z_STREAM_END) {
			if (selfAlloc)
				raw = out;
			rawLen = in_s.total_out;
			return Error();
		} else if (r == Z_BUF_ERROR || r == Z_OK) {
			if (selfAlloc) {
				if (tryLen == rawLen) {
					free(out);
					return Error("Not enough space to decompress zlib buffer");
				}
				tryLen = std::min(maxRawLen, tryLen * 2);
				continue;
			} else {
				return Error("Not enough space to decompress zlib buffer");
			}
		} else {
			if (selfAlloc)
				free(out);
			return Error::Fmt("Error decompressing zlib data: %v", r);
		}
	}

	// unreachable
	return Error();
}

IMQS_PAL_API Error DeflateInit(z_stream& stream, uint32_t flags) {
	if (!(flags & (FlagDeflate | FlagGzip)))
		return Error("Must specify either FlagDeflate or FlagGzip");

	int  level  = LevelFromFlags(flags);
	bool isGZip = !!(flags & FlagGzip);
	memset(&stream, 0, sizeof(stream));
	int r = deflateInit2(&stream, level, Z_DEFLATED, isGZip ? 16 + 15 : 15, 8, Z_DEFAULT_STRATEGY);
	if (r != Z_OK)
		return Error::Fmt("deflateInit error %v", r);

	return Error();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Compressor::Compressor() {
}

Compressor::Compressor(io::Writer* target, uint32_t flags) : Target(target), Flags(flags) {
}

Compressor::~Compressor() {
	if (IsInitialized)
		deflateEnd(&Stream);
}

void Compressor::SetFlags(uint32_t flags) {
	IMQS_ASSERT(!IsInitialized);
	Flags = flags;
}

Error Compressor::Write(const void* buf, size_t len) {
	IMQS_ASSERT(len < MaxZLibDataSize);

	if (!IsInitialized) {
		auto err = DeflateInit(Stream, Flags);
		if (!err.OK())
			return err;
		IsInitialized = true;
	}

	// Zero research behind these numbers
	if (TotalCompressed > 256 * 1024)
		OutBuf.Ensure(65536);
	else
		OutBuf.Ensure(4096);

	Stream.avail_in = (uInt) len;
	Stream.next_in  = (Bytef*) buf;
	bool isFlush    = len == 0;

	while (Stream.avail_in != 0 || isFlush) {
		Stream.avail_out = (uInt) OutBuf.Cap;
		Stream.next_out  = OutBuf.Buf;

		int r = deflate(&Stream, isFlush ? Z_FINISH : Z_NO_FLUSH);
		if (r != Z_OK && r != Z_STREAM_END)
			return Error::Fmt("deflate error %v", Stream.msg);

		if (Stream.next_out != OutBuf.Buf) {
			size_t csize = OutBuf.Cap - Stream.avail_out;
			TotalCompressed += csize;
			auto err = Target->Write(OutBuf.Buf, csize);
			if (!err.OK())
				return err;
		}

		// flush is complete
		if (r == Z_STREAM_END)
			break;
	}
	return Error();
}

} // namespace zlib
} // namespace compress
} // namespace imqs
