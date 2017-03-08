#pragma once

#include "../Math_.h"

namespace imqs {
namespace geom2d {

template <typename T>
class BBox2 {
public:
	T X1 = imqs::math::Traits<T>::Max();
	T Y1 = imqs::math::Traits<T>::Max();
	T X2 = -imqs::math::Traits<T>::Max();
	T Y2 = -imqs::math::Traits<T>::Max();

	BBox2() {
	}

	void Reset() {
		*this = BBox2();
	}

	void ExpandToFit(T x, T y) {
		X1 = std::min(X1, x);
		Y1 = std::min(Y1, y);
		X2 = std::max(X2, x);
		Y2 = std::max(Y2, y);
	}

	// Returns true if the boxes overlap or touch.
	bool PositiveUnion(const BBox2& b) const {
		return b.X2 >= X1 && b.X1 <= X2 &&
		       b.Y2 >= Y1 && b.Y1 <= Y2;
	}

	// Returns true if the other box is inside or equal to this box.
	bool IsInsideMe(const BBox2& b) const {
		return b.X1 >= X1 && b.Y1 >= Y1 &&
		       b.X2 <= X2 && b.Y2 <= Y2;
	}

	T Width() const {
		return X2 - X1;
	}

	T Height() const {
		return Y2 - Y1;
	}
};

typedef BBox2<double> BBox2d;
typedef BBox2<float>  BBox2f;
}
}