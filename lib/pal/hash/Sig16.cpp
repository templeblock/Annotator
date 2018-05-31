#include "pch.h"
#include "Sig16.h"
#include "SpookyV2.h"
#include "../strings/strings.h"

namespace imqs {
namespace hash {

void Sig16::MixBytes(const void* buf, size_t len) {
	SpookyHash::Hash128(buf, len, QWords, QWords + 1);
}

void Sig16::MixBytes(const std::string& str) {
	MixBytes(str.c_str(), str.size());
}

Sig16 Sig16::Compute(const void* buf, size_t len) {
	Sig16 s;
	s.MixBytes(buf, len);
	return s;
}

Sig16 Sig16::Compute(const std::string& str) {
	Sig16 s;
	s.MixBytes(str.c_str(), str.size());
	return s;
}

Sig16 Sig16::Combine(const Sig16& a, const Sig16& b) {
	Sig16 all[2] = {a, b};
	return Sig16::Compute(all, sizeof(all));
}

Sig16 Sig16::Combine(const Sig16& a, const Sig16& b, const Sig16& c) {
	Sig16 all[3] = {a, b, c};
	return Sig16::Compute(all, sizeof(all));
}

std::string Sig16::ToHex() const {
	char buf[Size * 2 + 1];
	strings::ToHex(this, Size, buf);
	buf[sizeof(buf) - 1] = 0;
	return buf;
}

} // namespace hash
} // namespace imqs
