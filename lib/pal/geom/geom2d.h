#pragma once

namespace imqs {
namespace geom2d {

IMQS_PAL_API double PolygonArea(size_t n, const double* pgon, size_t stride_in_doubles);
IMQS_PAL_API double PolygonArea(size_t n, const float* pgon, size_t stride_in_floats);

// 2D polygon orientation - clockwise, counterclockwise, or invalid
enum class PolyOrient {
	Invalid,
	CW,
	CCW,
};

// Checks whether a polygon is CCW or CW
IMQS_PAL_API PolyOrient PolygonOrient(size_t n, const double* pgon, size_t stride_in_doubles);
IMQS_PAL_API PolyOrient PolygonOrient(size_t n, const float* pgon, size_t stride_in_floats);

IMQS_PAL_API bool PtInsidePoly(double x, double y, size_t n, const double* v, size_t stride_in_doubles);
IMQS_PAL_API bool PtInsidePoly(float x, float y, size_t n, const float* v, size_t stride_in_floats);
IMQS_PAL_API bool PtInsidePoly(int x, int y, size_t n, const int* v, size_t stride_in_ints);

// Return on which side of the line (x1,y1)-(x2,y2), the point (px,py) lies.
// Use the sign of the return value to determine the side.
template <typename T>
T SideOfLine(T x1, T y1, T x2, T y2, T px, T py) {
	return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
}

} // namespace geom2d
} // namespace imqs
