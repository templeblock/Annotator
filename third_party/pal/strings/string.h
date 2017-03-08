#pragma once

namespace imqs {

/* A minimal std::string replacement.

The key thing that this String class guarantees is that it does not store 
pointers to internal memory. This gives the class the ability to be moved
around with memcpy. The same is not true for std::string. You cannot
reliably move std::string without using it's move constructor. In particular,
the VS 2015 std::string class has a fixed amount of internal storage for
small strings, and it will point at that internal storage when the string
is small enough.

Data inside this string is implicitly UTF-8.

This class is not intended to be used to manipulate strings. It is only
intended to be used to store them. One thing in particular is that every
access to a byte out of this string requires a branch, to check whether
Data is null. That is why we don't provide an operator[] - it's simply
too expensive. However, if you were to iterate over the contents of
this string using utfz, then there would be no loss in performance,
because you'd pass the utfz iterator a starting byte and a length.

The interface of this class is intentionally compatible with std::string,
so that if you need switch your implementation from one to the other, it
is as painless as possible.

Memory Layout

[pointer][pointer][....]
[  len  ][  cap  ][data]

In other words, sizeof(String) is equal to sizeof(void*).

len is the number of bytes in the string.
cap is the number of bytes allocated for string storage, including the null terminator. So cap for a string of length 1 is 2.

*/
class IMQS_PAL_API String {
public:
	String() : Data(nullptr) {}
	String(const String& s);
	String(String&& s);
	String(const char* s);
	String(const std::string& s);
	~String();

	String&     operator=(const String& s);
	String&     operator=(String&& s);
	String&     operator=(const char* s);
	String&     operator=(const std::string& s);
	String&     operator+=(char c);
	const char* c_str() const;
	size_t      length() const;
	uint32_t    hash32() const;

	operator std::string() const;

	bool operator==(const String& s) const;
	bool operator!=(const String& s) const;
	bool operator==(const char* s) const;
	bool operator!=(const char* s) const;

private:
	char* Data;

	void          Set(const char* s, size_t len);
	void          Alloc(size_t len);
	void          InitialAllocForGrowth();
	const char*   StrPtr() const;
	char*         StrPtr();
	const size_t& Len() const;
	size_t&       Len();
	const size_t& Cap() const;
	size_t&       Cap();
};
}

namespace ohash {
template <>
inline hashkey_t gethashcode(const imqs::String& k) {
	return k.hash32();
}
}