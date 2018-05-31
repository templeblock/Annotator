#include "pch.h"
#include "BitVector.h"
#include "../alloc.h"

namespace imqs {

BitVector::BitVector() {
}

BitVector::BitVector(const BitVector& b) {
	*this = b;
}

BitVector::BitVector(BitVector&& b) {
	*this = std::move(b);
}

BitVector::~BitVector() {
	free(Buf);
}

BitVector& BitVector::operator=(const BitVector& b) {
	if (WordCap != b.WordCap) {
		free(Buf);
		Buf     = nullptr;
		WordCap = b.WordCap;
		if (WordCap)
			Buf = (TWord*) imqs_malloc_or_die(WordCap * sizeof(TWord));
	}
	_Size = b._Size;
	// This "if (WordCap)" check is here to satisfy MSVC /analyze, but shouldn't be necessary
	if (WordCap) {
		size_t sizeBytes = (_Size + 7) / 8;
		memcpy(Buf, b.Buf, sizeBytes);
	}
	return *this;
}

BitVector& BitVector::operator=(BitVector&& b) {
	free(Buf);
	Buf     = b.Buf;
	WordCap = b.WordCap;
	_Size   = b._Size;

	b.Buf     = nullptr;
	b.WordCap = 0;
	b._Size   = 0;
	return *this;
}

void BitVector::Clear() {
	free(Buf);
	Buf     = nullptr;
	WordCap = 0;
	_Size   = 0;
}

void BitVector::Grow() {
	size_t newcap = std::max(WordCap * 2, 64 / sizeof(TWord));
	Buf           = (TWord*) imqs_realloc_or_die(Buf, newcap * sizeof(TWord));
	WordCap       = newcap;
}

BitVector::Mixture BitVector::ScanAll() const {
	if (_Size == 0)
		return Mixture::Empty;
	size_t nWhole  = _Size / TWordBits;
	bool   hasOne  = false;
	bool   hasZero = false;
	for (size_t i = 0; i < nWhole; i++) {
		if (Buf[i] == 0) {
			hasZero = true;
		} else if (Buf[i] == (TWord) -1) {
			hasOne = true;
		} else {
			hasZero = true;
			hasOne  = true;
		}

		if (hasZero && hasOne)
			break;
	}
	// scan the last remaining bits that didn't fit into a whole word
	for (size_t i = nWhole * TWordBits; i < _Size; i++) {
		if (!Get(i))
			hasZero = true;
		else
			hasOne = true;
		if (hasZero && hasOne)
			break;
	}
	if (hasZero && hasOne)
		return Mixture::Mixed;
	return hasOne ? Mixture::AllOne : Mixture::AllZero;
}

} // namespace imqs
