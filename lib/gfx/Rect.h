#pragma once

namespace imqs {
namespace gfx {

template <typename T>
class Rect {
public:
	T x1;
	T y1;
	T x2;
	T y2;

	Rect() {}
	Rect(T x1, T y1, T x2, T y2) : x1(x1), y1(y1), x2(x2), y2(y2) {}

	T Width() const { return x2 - x1; }
	T Height() const { return y2 - y1; }
};

typedef Rect<int32_t> Rect32;
typedef Rect<int64_t> Rect64;

} // namespace gfx
} // namespace imqs