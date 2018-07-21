#pragma once

namespace imqs {
namespace gfx {

template <typename T>
class Point {
public:
	T x;
	T y;

	Point() {}
	Point(T _x, T _y) : x(_x), y(_y) {}

	Point operator+(const Point& p) const { return Point(x + p.x, y + p.y); }
	Point operator-(const Point& p) const { return Point(x - p.x, y - p.y); }
};

typedef Point<int32_t> Point32;
typedef Point<int64_t> Point64;

} // namespace gfx
} // namespace imqs