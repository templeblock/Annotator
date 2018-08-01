#pragma once

namespace imqs {
namespace gfx {

// Compute control points for the curve running between points 0,1,2, with a smoothness factor.
// A smoothness factor of 0 produces straight lines. A typical smoothness factor is from 0.2 to 0.5.
// From http://scaledinnovation.com/analytics/splines/aboutSplines.html
template <typename T>
void ComputeSmoothCubicBezierControlPoints(T x0, T y0, T x1, T y1, T x2, T y2, T smoothness, T& p1x, T& p1y, T& p2x, T& p2y) {
	T d01 = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
	T d12 = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
	T fa  = smoothness * d01 / (d01 + d12); // scaling factor for triangle Ta
	T fb  = smoothness * d12 / (d01 + d12); // ditto for Tb, simplifies to fb=t-fa
	p1x   = x1 - fa * (x2 - x0);            // x2-x0 is the width of triangle T
	p1y   = y1 - fa * (y2 - y0);            // y2-y0 is the height of T
	p2x   = x1 + fb * (x2 - x0);
	p2y   = y1 + fb * (y2 - y0);
}

template <typename TVec, typename FT>
TVec EvaluateCubicBezier(const TVec& p0, const TVec& p1, const TVec& p2, const TVec& p3, FT t) {
	FT a = pow((FT) 1 - t, (FT) 3);
	FT b = (FT) 3 * pow((FT) 1 - t, (FT) 2) * t;
	FT c = (FT) 3 * ((FT) 1 - t) * t * t;
	FT d = pow(t, (FT) 3);
	return a * p0 + b * p1 + c * p2 + d * p3;
}

} // namespace gfx
} // namespace imqs