#pragma once

namespace imqs {
namespace geom2d {

/*
Benchmark Core i7, 24 bits of precision:

32-bit
------
  Sierpinski: 666 clocks
     Hilbert: 207 clocks
      Morton: 330 clocks

x64
---
  Sierpinski: 599 clocks
     Hilbert: 170 clocks
      Morton: 120 clocks

*/

IMQS_PAL_API uint64_t SierpinskiIndex(uint32_t max_xy, uint32_t x, uint32_t y);
IMQS_PAL_API uint64_t HilbertIndex(uint32_t bits, uint32_t x, uint32_t y);

template <typename TResult, uint32_t bits>
TResult MortonOrder(uint32_t x, uint32_t y) {
	TResult  r     = 0;
	uint32_t bit   = 1;
	uint32_t shift = 0;
	for (uint32_t b = bits; b != 0; b--) {
		r = r | (((TResult) x & bit) << shift);
		shift++;
		r = r | (((TResult) y & bit) << shift);
		bit <<= 1;
	}
	return r;
}

} // namespace geom2d
} // namespace imqs