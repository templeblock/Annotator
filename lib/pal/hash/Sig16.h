#pragma once

namespace imqs {
namespace hash {

// Signature of a tile, which is used to identify a tile in the tile cache.
// A tile's signature is a hash of everything that goes into generating it, including
// the rendering rules that determine the line thickness, colors, etc, as well as
// a hashed signature of the data that was used to generate the tile. This data
// is typically one or more database tables.
//
// Is 16 bytes of hash enough (Search for "Birthday Attack")?
// P(collision) = 1 - e^(-n^2 / 2*H), where H = 2^hashbits.
// For a 100 TB tile cache, with average tile size 10000 bytes, we have 10 billion tiles,
// so let's say we want to target n = 1e10 (ie 10 billion).
// Then, probability of a collision with a 16 byte (128 bit) hash = 1.5e-19.
// I think that's OK.
class IMQS_PAL_API Sig16 {
public:
	static const int Size      = 16;
	uint64_t         QWords[2] = {0};

	Sig16() {}
	Sig16(uint64_t q1, uint64_t q2) {
		QWords[0] = q1;
		QWords[1] = q2;
	}

	// Equivalent to `Signature s; s.MixBytes(buf, len); return s;`
	static Sig16 Compute(const void* buf, size_t len);
	static Sig16 Compute(const std::string& str);
	static Sig16 Combine(const Sig16& a, const Sig16& b);
	static Sig16 Combine(const Sig16& a, const Sig16& b, const Sig16& c);

	void MixBytes(const void* buf, size_t len); // Hash 'buf', and mix it into this Signature
	void MixBytes(const std::string& str);      // Hash 'str', and mix it into this Signature

	bool operator==(const Sig16& s) const { return memcmp(QWords, s.QWords, Size) == 0; }
	bool operator!=(const Sig16& s) const { return !(*this == s); }

	std::string ToHex() const;
};

} // namespace hash
} // namespace imqs

namespace ohash {
template <>
inline hashkey_t gethashcode(const imqs::hash::Sig16& s) {
	static_assert(imqs::hash::Sig16::Size == 16, "Four XORs is no longer correct");
	const uint32_t* w = (const uint32_t*) s.QWords;
	return w[0] ^ w[1] ^ w[2] ^ w[3];
};
} // namespace ohash

namespace std {
template <>
struct hash<imqs::hash::Sig16> {
	size_t operator()(const imqs::hash::Sig16& s) const {
#ifdef IMQS_ARCH_32
		const uint32_t* w = (const uint32_t*) s.QWords;
		return w[0] ^ w[1] ^ w[2] ^ w[3];
#else
		return s.QWords[0] ^ s.QWords[1];
#endif
	}
};
} // namespace std
