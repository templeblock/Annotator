#include "pch.h"
#include "geom2d.h"

namespace imqs {
namespace geom2d {

template <typename TReal>
double TSignedPolygonArea(size_t n, const TReal* pgon, size_t stride) {
	double area = 0;

	const TReal* tail = pgon + (n - 1) * stride;
	const TReal* head = pgon;
	for (; n != 0; n--) {
		double t = 0;
		t += ((double) tail[0] - (double) pgon[0]) * ((double) head[1] - (double) pgon[1]); // x * y
		t -= ((double) tail[1] - (double) pgon[1]) * ((double) head[0] - (double) pgon[0]); // y * x
		area += t;
		tail = head;
		head += stride;
	}

	area *= 0.5;
	return area;
}

double PolygonArea(size_t n, const double* pgon, size_t stride_in_doubles) {
	return fabs(TSignedPolygonArea(n, pgon, stride_in_doubles));
}

double PolygonArea(size_t n, const float* pgon, size_t stride_in_floats) {
	return fabs(TSignedPolygonArea(n, pgon, stride_in_floats));
}

template <typename TReal>
PolyOrient TPolygonOrient(size_t n, const TReal* pgon, size_t stride) {
#define X(i) pgon[(i) *stride + 0]
#define Y(i) pgon[(i) *stride + 1]

	if (n < 3)
		return PolyOrient::Invalid;

	// Use method described in comp.graphics.algorithms faq 2.07
	size_t rightlow = 0;
	for (size_t i = 1; i < n; i++) {
		if ((Y(i) < Y(rightlow)) ||
		    (Y(i) == Y(rightlow) && X(i) > X(rightlow))) {
			rightlow = i;
		}
	}

	size_t vleft  = (rightlow + n - 1) % n;
	size_t vright = (rightlow + 1) % n;

	/*
	a = pgon[ vleft ];
	b = pgon[ rightlow ];
	c = pgon[ vright ];

	We translate everything so that a.x, a.y is the origin. This helps with precision when the coordinates are large.

	Natural:
	double area = a.x * b.y - a.y * b.x   +   a.y * c.x - a.x * c.y   +   b.x * c.y - c.x * b.y;

	Shifted: (a.x and a.y become zero)
	*/
	TReal area = (X(rightlow) - X(vleft)) * (Y(vright) - Y(vleft)) - (X(vright) - X(vleft)) * (Y(rightlow) - Y(vleft));

	if (area > 0)
		return PolyOrient::CCW;
	else if (area < 0)
		return PolyOrient::CW;
	else
		return PolyOrient::Invalid;

#undef X
#undef Y
}

PolyOrient PolygonOrient(size_t n, const float* pgon, size_t stride_in_floats) {
	return TPolygonOrient(n, pgon, stride_in_floats);
}

PolyOrient PolygonOrient(size_t n, const double* pgon, size_t stride_in_doubles) {
	return TPolygonOrient(n, pgon, stride_in_doubles);
}

template <typename TReal>
bool TPtInsidePoly(TReal x, TReal y, size_t n, const TReal* v, size_t stride) {
	size_t       i, j, c = 0;
	size_t       n_strided = n * stride;
	const TReal* vx        = v;
	const TReal* vy        = v + 1;
	for (i = 0, j = (n - 1) * stride; i < n_strided;) {
		if ((
		        ((vy[i] <= y) && (y < vy[j])) ||
		        ((vy[j] <= y) && (y < vy[i]))) &&
		    (x < (vx[j] - vx[i]) * (y - vy[i]) / (vy[j] - vy[i]) + vx[i]))
			c ^= 1;
		j = i;
		i += stride;
	}
	return c != 0;
}

bool PtInsidePoly(double x, double y, size_t n, const double* v, size_t stride_in_doubles) {
	return TPtInsidePoly(x, y, n, v, stride_in_doubles);
}

bool PtInsidePoly(float x, float y, size_t n, const float* v, size_t stride_in_floats) {
	return TPtInsidePoly(x, y, n, v, stride_in_floats);
}
}
}
