#include "pch.h"
#include "path.h"

namespace imqs {
namespace path {

static size_t LastSep(const char* p) {
	size_t i    = 0;
	size_t last = -1;
	for (; p[i]; i++) {
		if (IsSeparator(p[i]))
			last = i;
	}
	return last;
}

IMQS_PAL_API bool IsSeparator(int ch) {
#ifdef _WIN32
	return ch == '/' || ch == '\\';
#else
	return ch == '/';
#endif
}

IMQS_PAL_API std::string Dir(const char* path) {
	size_t lastSep = LastSep(path);
	if (lastSep == -1)
		return path;

	// Always return return / as parent directory of /x, or /
	if (lastSep == 0)
		return std::string(path, 1);

	return std::string(path, lastSep);
}
}
}
