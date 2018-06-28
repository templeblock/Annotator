#pragma once

namespace imqs {
namespace gfx {
namespace raster {

inline uint32_t Bilinear_x64(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t ix, uint32_t iy) {
	// By Nils Pipenbrinck.
	uint64_t mask = 0x00ff00ff00ff00ffLL;
	uint64_t a64  = (a | ((uint64_t) a << 24)) & mask;
	uint64_t b64  = (b | ((uint64_t) b << 24)) & mask;
	uint64_t c64  = (c | ((uint64_t) c << 24)) & mask;
	uint64_t d64  = (d | ((uint64_t) d << 24)) & mask;
	a64           = a64 + (((b64 - a64) * ix + mask) >> 8);
	c64           = c64 + (((d64 - c64) * ix + mask) >> 8);
	a64 &= mask;
	c64 &= mask;
	a64 = a64 + (((c64 - a64) * iy + mask) >> 8);
	a64 &= mask;
	return (uint32_t)(a64 | (a64 >> 24));
}

// Perform bilinear filtering on a 4 channel image. u and v are base-256
// u and v are first clamped to 0..uclamp and 0..vclamp
inline uint32_t ImageBilinear(const void* src, int width, int32_t uclamp, int32_t vclamp, int32_t u, int32_t v) {
	if (u < 0)
		u = 0;
	if (u >= uclamp)
		u = uclamp;
	if (v < 0)
		v = 0;
	if (v >= vclamp)
		v = vclamp;
	uint32_t        iu   = (uint32_t) u >> 8;
	uint32_t        iv   = (uint32_t) v >> 8;
	uint32_t        ru   = (uint32_t) u & 0xff;
	uint32_t        rv   = (uint32_t) v & 0xff;
	const uint32_t* src1 = (const uint32_t*) src + iv * width + iu;
	const uint32_t* src2 = (const uint32_t*) src + (iv + 1) * width + iu;
	return Bilinear_x64(src1[0], src1[1], src2[0], src2[1], ru, rv);
}

// Perform bilinear filtering on a 4 channel image. u and v are base-256
inline uint32_t ImageBilinearUnclamped(const void* src, int width, int32_t u, int32_t v) {
	uint32_t        iu   = (uint32_t) u >> 8;
	uint32_t        iv   = (uint32_t) v >> 8;
	uint32_t        ru   = (uint32_t) u & 0xff;
	uint32_t        rv   = (uint32_t) v & 0xff;
	const uint32_t* src1 = (const uint32_t*) src + iv * width + iu;
	const uint32_t* src2 = (const uint32_t*) src + (iv + 1) * width + iu;
	return Bilinear_x64(src1[0], src1[1], src2[0], src2[1], ru, rv);
}

} // namespace raster
} // namespace gfx
} // namespace imqs