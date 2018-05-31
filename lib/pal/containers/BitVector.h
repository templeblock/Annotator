#pragma once

namespace imqs {

// Forcing Set() to be inline takes it's speed down from 2.4ns to 1.4ns
#ifdef _MSC_VER
#define BITVECTOR_INLINE __forceinline
#else
#define BITVECTOR_INLINE inline
#endif

// This is identical in spirit to vector<bool>, but it is much faster than vector<bool>
// in debug builds on MSVC. This was created mostly for that reason - ie debug builds
// on MSVC. In addition, this also allows us to create functions such as IsAllZero(),
// much more efficiently than vector<bool>
//
//
// Benchmarks      Timings for Release/Debug
// ---------------------------------------------------
// Operation       BitVector    MSVC 2015 vector<bool>
// Add 1 bit       1.9/30 ns    37/1140 ns
// Set 1 bit       1.5/30 ns    1.4/467 ns
//
// MSVC's vector<bool> is competitive on Set(), in a release build, but on everything else
// it's miserably slow.
class IMQS_PAL_API BitVector {
public:
	typedef uint64_t    TWord;
	static const size_t TWordBits = sizeof(TWord) * 8;

	enum class Mixture {
		Empty,
		AllZero,
		AllOne,
		Mixed,
	};

	BitVector();
	BitVector(const BitVector& b);
	BitVector(BitVector&& b);
	~BitVector();

	BitVector& operator=(const BitVector& b);
	BitVector& operator=(BitVector&& b);

	void Clear();

	size_t Size() const { return _Size; }

	void    Add(bool b);
	void    Set(size_t i, bool b);
	bool    Get(size_t i) const;
	Mixture ScanAll() const;

	bool operator[](size_t i) const { return Get(i); }

private:
	size_t _Size   = 0; // Size in bits
	size_t WordCap = 0; // Capacity in TWords
	TWord* Buf     = nullptr;

	void Grow();
};

BITVECTOR_INLINE void BitVector::Add(bool b) {
	size_t   word   = _Size / (TWordBits);
	unsigned offset = (unsigned) (_Size % TWordBits);
	if (word >= WordCap)
		Grow();

	// This non-branch formulation seems to be a tiny bit slower on my i7 4700
	//TWord clear = ~((TWord) 1 << offset);
	//TWord bit   = ((TWord) b) << offset;
	//Buf[word]   = (Buf[word] & clear) | bit;

	if (b)
		Buf[word] = Buf[word] | ((TWord) 1 << offset);
	else
		Buf[word] = Buf[word] & ~((TWord) 1 << offset);
	_Size++;
}

BITVECTOR_INLINE void BitVector::Set(size_t i, bool b) {
	IMQS_ASSERT(i < _Size);
	size_t   word   = i / TWordBits;
	unsigned offset = (unsigned) (i % TWordBits);
	if (b)
		Buf[word] = Buf[word] | ((TWord) 1 << offset);
	else
		Buf[word] = Buf[word] & ~((TWord) 1 << offset);
}

BITVECTOR_INLINE bool BitVector::Get(size_t i) const {
	IMQS_ASSERT(i < _Size);
	size_t   word   = i / TWordBits;
	unsigned offset = (unsigned) (i % TWordBits);
	TWord    bit    = (TWord) 1 << offset;
	return !!(Buf[word] & bit);
}

} // namespace imqs
