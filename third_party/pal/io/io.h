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
}
}
