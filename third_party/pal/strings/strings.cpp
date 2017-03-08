#include "pch.h"
#include "strings.h"

namespace imqs {

IMQS_PAL_API int64_t AtoI64(const char* s) {
#ifdef _MSC_VER
	return _atoi64(s);
#else
	return (int64_t) atoll(s);
#endif
}

// This is from http://www.jb.man.ac.uk/~slowe/cpp/itoa.html
template <typename TINT, typename TCH>
size_t ItoAT(TINT value, TCH* result, int base) {
	if (base < 2 || base > 36) {
		*result = '\0';
		return 0;
	}

	TCH *ptr = result, *ptr1 = result, tmp_char;
	TINT tmp_value;

	do {
		tmp_value = value;
		value /= base;
		*ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz"[35 + (tmp_value - value * base)];
	} while (value);

	// Apply negative sign
	if (tmp_value < 0)
		*ptr++ = '-';
	size_t written = (size_t)(ptr - result);
	*ptr--         = '\0';
	while (ptr1 < ptr) {
		tmp_char = *ptr;
		*ptr--   = *ptr1;
		*ptr1++  = tmp_char;
	}
	return written;
}

IMQS_PAL_API size_t ItoA(int value, char* result, int base) {
	return ItoAT(value, result, base);
}

IMQS_PAL_API size_t I64toA(int64_t value, char* result, int base) {
	return ItoAT(value, result, base);
}

namespace strings {
IMQS_PAL_API void ToHex(const void* buf, size_t len, char* out) {
	const char* lut = "0123456789ABCDEF";
	auto        src = (const uint8_t*) buf;
	for (; len != 0; len--, src++, out += 2) {
		out[0] = lut[*src >> 4];
		out[1] = lut[*src & 15];
	}
	*out = 0;
}

IMQS_PAL_API std::string tolower(const std::string& s) {
	std::string low;
	low.reserve(s.length());
	for (int cp : utfz::cp(s))
		utfz::encode(low, ::tolower(cp));
	return low;
}

IMQS_PAL_API std::string toupper(const std::string& s) {
	std::string up;
	up.reserve(s.length());
	for (int cp : utfz::cp(s))
		utfz::encode(up, ::toupper(cp));
	return up;
}

IMQS_PAL_API bool EndsWith(const std::string& s, const char* suffix) {
	auto pos = s.rfind(suffix, 0);
	if (pos == -1)
		return false;
	return pos == s.length() - strlen(suffix);
}

IMQS_PAL_API std::string TrimRight(const std::string& s) {
	size_t i = s.length() - 1;
	for (; i != -1; i--) {
		char c = s[i];
		if (!(c == ' ' || c == '\t' || c == '\n' || c == '\r'))
			break;
	}
	return s.substr(0, i + 1);
}

template <bool CaseSensitive>
bool MatchWildcardT(const char* s, const char* p) {
	if (*p == '*') {
		while (*p == '*')
			++p;
		if (*p == 0)
			return true;
		while (*s != 0 && !MatchWildcardT<CaseSensitive>(s, p)) {
			int cp;
			if (!utfz::next(s, cp))
				break;
		}
		return *s != 0;
	} else if (*p == 0 || *s == 0) {
		return *p == *s;
	} else {
		int px = utfz::decode(p);
		int sx = utfz::decode(s);

		if ((CaseSensitive ? (px == sx) : ::tolower(px) == ::tolower(sx)) || *p == '?') {
			int cp;
			utfz::next(s, cp);
			utfz::next(p, cp);
			return MatchWildcardT<CaseSensitive>(s, p);
		} else {
			return false;
		}
	}
}

IMQS_PAL_API bool MatchWildcard(const std::string& s, const std::string& p) {
	return MatchWildcardT<true>(s.c_str(), p.c_str());
}
IMQS_PAL_API bool MatchWildcard(const char* s, const char* p) {
	return MatchWildcardT<true>(s, p);
}
IMQS_PAL_API bool MatchWildcardNoCase(const std::string& s, const std::string& p) {
	return MatchWildcardT<false>(s.c_str(), p.c_str());
}
IMQS_PAL_API bool MatchWildcardNoCase(const char* s, const char* p) {
	return MatchWildcardT<false>(s, p);
}

IMQS_PAL_API std::string Join(const std::vector<std::string>& parts, const char* joiner) {
	std::string r;
	for (size_t i = 0; i < parts.size(); i++) {
		r += parts[i];
		if (i != parts.size() - 1)
			r += joiner;
	}
	return r;
}
}
}
