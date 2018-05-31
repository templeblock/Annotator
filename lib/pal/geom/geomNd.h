#pragma once

#include <limits>
#include <cmath>
#include <cstdlib>

namespace imqs {
namespace geom {

// Snap pt to the line that runs through P1..P2
// If isSeg is true, then treat P1..P2 as a line segment.
// If isSeg is false, then treat P1..P2 as an infinitely long line.
// If mu is provided, then it is filled with the location along P1..P2, where 0 is at P1, and 1 is at P2.
template <typename TVec>
TVec ClosestPtOnLineT(const TVec& pt, const TVec& P1, const TVec& P2, bool isSeg, typename TVec::FT* mu = nullptr) {
	typedef typename TVec::FT FT;
	if (pt == P1) {
		if (mu)
			*mu = 0;
		return P1;
	}
	if (pt == P2) {
		if (mu)
			*mu = 1;
		return P2;
	}
	TVec me   = P2 - P1;
	TVec sub2 = pt - P1;
	FT   L    = me.dot(me);
	if (std::abs(L) < std::numeric_limits<FT>::epsilon()) {
		if (mu)
			*mu = 0;
		return P1;
	}
	FT r = sub2.dot(me) / L;
	if (mu)
		*mu = r;
	if (!isSeg || (r >= 0 && r <= 1))
		return P1 + r * me;
	else {
		if (r < 0)
			return P1;
		else
			return P2;
	}
}

// Snaps pt onto the nearest location on line, and then returns that position.
// Also returns the fractional position of that location on the line, in the parameter mu.
// Our fractional system makes integer locations correspond to exact vertices. The fractional part
// of the position specifies a relative position on the segment on which the vertex lies.
// 'pt' is INOUT
// 'mu' is OUT
// Returns the distance from the origin 'pt' to the snapped position
template <typename TVec>
typename TVec::FT SnapPointToLine(bool isClosed, size_t nVertex, const TVec* line, TVec& pt, typename TVec::FT& mu) {
	typedef typename TVec::FT FT;
	if (nVertex < 2)
		return 0;

	size_t i   = 1;
	size_t h   = 0;
	size_t top = nVertex;
	if (isClosed) {
		h = nVertex - 1;
		i = 0;
	}

	FT     bestDistance = std::numeric_limits<FT>::max();
	TVec   bestPt;
	size_t bestsegH = 0;
	size_t bestsegI = 1;
	FT     bestMu   = 0;
	for (; i < top; h = i++) {
		double localMu;
		auto   closest  = ClosestPtOnLineT(pt, line[h], line[i], true, &localMu);
		auto   distance = closest.distance(pt);
		if (distance < bestDistance) {
			bestPt       = closest;
			bestDistance = distance;
			bestsegH     = h;
			bestsegI     = i;
			bestMu       = localMu;
		}
	}

	bestMu += (FT) bestsegH;
	// Clamp the returned mu, so that it is safe for the caller to do 'line[floor(mu)]'
	if (isClosed && bestMu >= nVertex) {
		bestMu = 0;
	} else if (!isClosed && bestMu >= nVertex - 1) {
		bestMu = (FT) nVertex;
	}
	pt = bestPt;
	mu = bestMu;
	return bestDistance;
}

// This is the 'opposite' of SnapPointToLine.
// Given a position on the line, return the interpolated position given by 'pos'.
template <typename TVec>
TVec ResolvePtOnLine(bool isClosed, size_t nVertex, const TVec* line, typename TVec::FT pos) {
	typedef typename TVec::FT FT;
	if (pos < 0) {
		pos = 0;
	} else if (!isClosed && pos >= (FT) nVertex - 1) {
		pos = FT(nVertex - 1);
	} else if (isClosed && pos >= (FT) nVertex) {
		pos = 0;
	}
	FT     iposF;
	FT     ifrac = std::modf(pos, &iposF);
	size_t ipos  = (size_t) iposF;
	if (ifrac == 0) {
		// exactly on a vertex
		return line[ipos];
	} else if (ipos != nVertex - 1) {
		// regular case
		return line[ipos] + ifrac * (line[ipos + 1] - line[ipos]);
	} else {
		// final segment of a closed polyline
		return line[ipos] + ifrac * (line[0] - line[ipos]);
	}
}

} // namespace geom
} // namespace imqs