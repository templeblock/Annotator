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
};

typedef Point<int32_t> Point32;

} // namespace gfx
} // namespace imqs