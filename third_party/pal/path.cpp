#include "pch.h"
#include "path.h"

namespace imqs {
namespace path {

#ifdef _WIN32
static const char Separator = '\\';
#else
static const char Separator = '/';
#endif

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

IMQS_PAL_API std::string Dir(const std::string& path) {
	size_t lastSep = LastSep(path.c_str());
	if (lastSep == -1)
		return path;

	// Always return return / as parent directory of /x, or /
	if (lastSep == 0)
		return std::string(path, 0, 1);

	return std::string(path, 0, lastSep);
}

IMQS_PAL_API std::string Filename(const std::string& path) {
	size_t lastSep = LastSep(path.c_str());
	if (lastSep == -1)
		return path;

	return path.substr(lastSep + 1);
}

IMQS_PAL_API std::string Join(const std::string& a, const std::string& b, const std::string& c, const std::string& d) {
	const std::string* parts[3] = {&b, &c, &d};
	size_t             n        = 1;
	if (c != "")
		n++;
	if (d != "")
		n++;

	std::string j = a;
	for (size_t i = 0; i < n; i++) {
		const auto& next = *parts[i];
		if (j.length() == 0 || !IsSeparator(j[j.length() - 1]))
			j += Separator;

		if (next.length() != 0 && IsSeparator(next.at(0)))
			j += next.substr(1);
		else
			j += next;
	}

	return j;
}
}
} // namespace imqs
