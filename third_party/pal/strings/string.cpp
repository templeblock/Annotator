#include "pch.h"
#include "string.h"
#include "../alloc.h"
#include "../hash/FNV1a.h"
#include "../hash/xxhash.h"

namespace imqs {

String::String(const String& s) {
	Data  = nullptr;
	*this = s;
}

String::String(String&& s) {
	Data   = s.Data;
	s.Data = nullptr;
}

String::String(const char* s) {
	Data  = nullptr;
	*this = s;
}

String::String(const std::string& s) {
	Data  = nullptr;
	*this = s;
}

String::~String() {
	free(Data);
}

String& String::operator=(const String& s) {
	if (s.Data)
		Set(s.StrPtr(), s.Len());
	else
		Set(nullptr, 0);
	return *this;
}

String& String::operator=(String&& s) {
	Data   = s.Data;
	s.Data = nullptr;
	return *this;
}

String& String::operator=(const char* s) {
	Set(s, strlen(s));
	return *this;
}

String& String::operator=(const std::string& s) {
	Set(s.c_str(), s.length());
	return *this;
}

String& String::operator+=(char c) {
	if (Data == nullptr)
		InitialAllocForGrowth();
	size_t len = Len();
	if (len + 1 == Cap())
		Alloc(len * 2);
	StrPtr()[len]     = c;
	StrPtr()[len + 1] = 0;
	Len()++;
	return *this;
}

const char* String::c_str() const {
	return StrPtr();
}

size_t String::length() const {
	if (Data)
		return Len();
	else
		return 0;
}

uint32_t String::hash32() const {
	if (!Data)
		return 0;

	// See http://aras-p.info/blog/2016/08/09/More-Hash-Function-Tests/

	size_t len = Len();
	auto   str = StrPtr();
	if (len <= 8)
		return fnv_32a_buf(str, len);
	else
		return XXH32(str, (uint32_t) len, 0);
}

String::operator std::string() const {
	if (!Data)
		return std::string();
	return std::string(StrPtr(), Len());
}

bool String::operator==(const String& s) const {
	if ((Data == nullptr) != (s.Data == nullptr))
		return false;
	if (!Data)
		return true;
	if (Len() != s.Len())
		return false;
	return memcmp(StrPtr(), s.StrPtr(), Len()) == 0;
}

bool String::operator!=(const String& s) const {
	return !(*this == s);
}

bool String::operator==(const char* s) const {
	if (!Data)
		return s[0] == 0;

	const char* str = StrPtr();
	size_t      len = Len();
	size_t      i   = 0;
	for (; i < len && s[i] != 0; i++) {
		if (s[i] != str[i])
			return false;
	}

	return i == len && s[i] == 0;
}

bool String::operator!=(const char* s) const {
	return !(*this == s);
}

void String::Set(const char* s, size_t len) {
	if (Data) {
		free(Data);
		Data = nullptr;
	}

	if (len != 0) {
		Alloc(len);
		Len() = len;
		memcpy(StrPtr(), s, len);
		StrPtr()[len] = 0;
	}
}

void String::Alloc(size_t len) {
	Data  = (char*) imqs_realloc_or_die(Data, 2 * sizeof(void*) + len + 1);
	Cap() = len + 1;
}

void String::InitialAllocForGrowth() {
	Alloc(8);
	Len()       = 0;
	StrPtr()[0] = 0;
}

const char* String::StrPtr() const {
	if (Data)
		return Data + 2 * sizeof(void*);
	else
		return nullptr;
}

char* String::StrPtr() {
	if (Data)
		return Data + 2 * sizeof(void*);
	else
		return nullptr;
}

const size_t& String::Len() const {
	return *((size_t*) Data);
}

size_t& String::Len() {
	return *((size_t*) Data);
}

const size_t& String::Cap() const {
	return *((size_t*) (Data + sizeof(void*)));
}

size_t& String::Cap() {
	return *((size_t*) (Data + sizeof(void*)));
}
}