#pragma once

namespace imqs {
namespace gfx {

template <typename TVec, typename TDistance>
void SimplifyR(TDistance distanceSQ, size_t n, const TVec* v, bool* keep) {
	if (n <= 2)
		return;
	auto      p1   = v[0];
	auto      p2   = v[n - 1];
	size_t    maxI = 0;
	TDistance maxD = 0;
	for (size_t i = 1; i < n - 1; i++) {
		auto distance = geom::ClosestPtOnLineT(v[i], p1, p2, true).distanceSQ(v[i]);
		if ((TDistance) distance > maxD) {
			maxI = i;
			maxD = distance;
		}
	}
	if (maxD < distanceSQ)
		return;
	keep[maxI] = true;
	SimplifyR(distanceSQ, maxI + 1, v, keep);
	SimplifyR(distanceSQ, n - maxI + 1, v + maxI, keep + maxI);
}

// Ramer Douglas Peucker simplifier.
// The output array must already be allocated, and must be at least nIn large.
template <typename TVec, typename TDistance>
void Simplify(TDistance distance, size_t nIn, const TVec* in, size_t& nOut, TVec* out) {
	bool* keep = new bool[nIn];
	memset(keep, 0, nIn);
	keep[0]       = true;
	keep[nIn - 1] = true;
	SimplifyR(distance * distance, nIn, in, keep);
	size_t j = 0;
	for (size_t i = 0; i < nIn; i++) {
		if (keep[i])
			out[j++] = in[i];
	}
	nOut = j;
	delete[] keep;
}

// A variant of the Ramer Douglas Peucker simplifier that only returns the 'keep' array, which is parallel to the input vertices
// The 'keep' array must be nIn large.
template <typename TVec, typename TDistance>
void Simplify(TDistance distance, size_t nIn, const TVec* in, bool* keep) {
	memset(keep, 0, nIn);
	keep[0]       = true;
	keep[nIn - 1] = true;
	SimplifyR(distance * distance, nIn, in, keep);
}

} // namespace gfx
} // namespace imqs