#include "pch.h"
#include "SpaceFillingCurve.h"

namespace imqs {
namespace geom2d {

IMQS_PAL_API uint64_t SierpinskiIndex(uint32_t max_xy, uint32_t x, uint32_t y) {
	uint32_t index = max_xy;
	uint64_t res   = 0;

	if (x > y) {
		res++;
		x = max_xy - x;
		y = max_xy - y;
	}

	while (index > 0) {
		res += res;
		if (x + y > max_xy) {
			res++;
			auto oldx = x;
			x         = max_xy - y;
			y         = oldx;
		}

		x += x;
		y += y;
		res += res;

		if (y > max_xy) {
			res++;
			auto oldx = x;
			x         = y - max_xy;
			y         = max_xy - oldx;
		}
		index = index >> 1;
	}

	return res;
}

IMQS_PAL_API uint64_t HilbertIndex(uint32_t bits, uint32_t x, uint32_t y) {
	uint64_t res = 0;
	x <<= 1; // by shifting one up first, we don't have to worry about the final shift, getting the LSB Y bit into bit position 2.
	y <<= 1;
	uint32_t bitx = bits;
	uint32_t bity = bits - 1;

	/*
	Position on Curve
	-----------------

	0      3
	|      |
	|      |
	1------2


	Types of Curves
	---------------

	0   |_|

	1   --+
		--+

	2   +--
		+--

	3   +-+
		| |
	*/

	uint32_t current     = 0;
	uint8_t  score[4][4] = {
        {1, 2, 0, 3},
        {3, 2, 0, 1},
        {1, 0, 2, 3},
        {3, 0, 2, 1},
    };
	uint8_t transition[4][4] = {
	    {0, 0, 1, 2},
	    {3, 1, 0, 1},
	    {2, 3, 2, 0},
	    {1, 2, 3, 3},
	};

	while (bitx) {
		uint32_t v = ((x >> bitx) & 1) | ((y >> bity) & 2);
		res        = (res << 2) | score[current][v];
		current    = transition[current][v];
		bitx--;
		bity--;
	}
	return res;
}

} // namespace geom2d
} // namespace imqs
