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

	static Rect Inverted() {
		return Rect(
		    std::numeric_limits<T>::max(),
		    std::numeric_limits<T>::max(),
		    std::numeric_limits<T>::lowest(),
		    std::numeric_limits<T>::lowest());
	}

	bool IsInverted() const {
		return x2 < x1 || y2 < y1;
	}

	T Width() const { return x2 - x1; }
	T Height() const { return y2 - y1; }

	void Expand(T x, T y) {
		x1 -= x;
		x2 += x;
		y1 -= y;
		y2 += y;
	}

	void Offset(T x, T y) {
		x1 += x;
		y1 += y;
		x2 += x;
		y2 += y;
	}

	void ExpandToFit(T x, T y) {
		x1 = std::min(x1, x);
		y1 = std::min(y1, y);
		x2 = std::max(x2, x);
		y2 = std::max(y2, y);
	}

	void ExpandToFit(const Rect& r) {
		if (r.IsInverted())
			return;
		ExpandToFit(r.x1, r.y1);
		ExpandToFit(r.x2, r.y2);
	}

	void CropTo(const Rect& r) {
		if (r.IsInverted())
			return;
		x1 = std::max(x1, r.x1);
		y1 = std::max(y1, r.y1);
		x2 = std::min(x2, r.x2);
		y2 = std::min(y2, r.y2);
	}
};

typedef Rect<int32_t> Rect32;
typedef Rect<int64_t> Rect64;
typedef Rect<float>   RectF;
typedef Rect<double>  RectD;

} // namespace gfx
} // namespace imqs