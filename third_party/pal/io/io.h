#pragma once

#include "../Error.h"

namespace imqs {

extern IMQS_PAL_API StaticError ErrEOF;

namespace io {

class IMQS_PAL_API Writer {
public:
	virtual ~Writer();
	virtual Error Write(const void* buf, size_t len) = 0;

	// Write formatted output
	template <typename... Args>
	Error Fmt(const char* fs, const Args&... args) {
		char buf[128];
		auto res = tsf::fmt_buf(buf, sizeof(buf), fs, args...);
		if (res.Len == 0)
			return Error();
		auto err = Write(buf, res.Len);
		if (res.Str != buf)
			delete[] res.Str;
		return err;
	}
};

class IMQS_PAL_API Reader {
public:
	virtual ~Reader();
	virtual Error Read(void* buf, size_t& len) = 0;
};

enum class SeekWhence {
	Begin,
	Current,
	End
};

class IMQS_PAL_API Seeker {
public:
	virtual ~Seeker();
	virtual Error SeekWithResult(int64_t offset, SeekWhence whence, int64_t& newPosition) = 0;

	// Seeks to end, then seeks back to current position.
	// If this function returns an error, then the seek position may not be where you left it.
	virtual Error Length(int64_t& len);

	Error Seek(int64_t offset, SeekWhence whence); // Calls Seek, then discards newPosition
};

} // namespace io
} // namespace imqs
