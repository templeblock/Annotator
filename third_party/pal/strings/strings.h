#pragma once

#include <utfz/utfz.h>

namespace imqs {

IMQS_PAL_API int64_t AtoI64(const char* s);
IMQS_PAL_API size_t ItoA(int value, char* result, int base);       // Returns size of string, excluding null terminator. Max 12 bytes written (including null terminator), for base 10
IMQS_PAL_API size_t I64toA(int64_t value, char* result, int base); // Returns size of string, excluding null terminator. Max 21 bytes written (including null terminator), for base 10

namespace strings {
IMQS_PAL_API void ToHex(const void* buf, size_t len, char* out);

IMQS_PAL_API std::string tolower(const std::string& s);
IMQS_PAL_API std::string toupper(const std::string& s);

IMQS_PAL_API bool EndsWith(const std::string& s, const char* suffix);

// Trim whitespace (space, tab, newline, carriage return) from the right end of the string
IMQS_PAL_API std::string TrimRight(const std::string& s);

IMQS_PAL_API bool MatchWildcard(const std::string& s, const std::string& p);
IMQS_PAL_API bool MatchWildcard(const char* s, const char* p);
IMQS_PAL_API bool MatchWildcardNoCase(const std::string& s, const std::string& p);
IMQS_PAL_API bool MatchWildcardNoCase(const char* s, const char* p);

// Split a UTF-8 string into a vector of strings
// delim: The splitting code point
template <typename TList>
void Split(const char* s, int delim, TList& list) {
	if (*s == 0)
		return;
	typedef typename TList::value_type TString;
	TString*                           item;
	list.push_back(TString());
	item = &list.back();
	for (auto cp : utfz::cp(s)) {
		if (cp == delim) {
			list.push_back(TString());
			item = &list.back();
		} else {
			char buf[5];
			int  nch = utfz::encode(buf, cp);
			for (int i = 0; i < nch; i++)
				*item += buf[i];
		}
	}
}

IMQS_PAL_API std::string Join(const std::vector<std::string>& parts, const char* joiner);
}
}
