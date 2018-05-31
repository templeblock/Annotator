#pragma once

#include "Vec3.h"

namespace imqs {
namespace gfx {

template <typename T>
class Mat3 {
public:
	typedef Vec3<T>     Vec3T;
	typedef Vec3Base<T> Vec3BaseT;

	Vec3Base<T> row[3];

	T& m(int _row, int col) { return row[_row][col]; }
	T  m(int _row, int col) const { return row[_row][col]; }

	void MakeIdentity() {
		row[0] = Vec3T(1, 0, 0);
		row[1] = Vec3T(0, 1, 0);
		row[2] = Vec3T(0, 0, 1);
	}

	// Natural format is row-major. This will make the matrix column-major
	void Transpose() {
		Vec3T col[4];
		col[0] = Vec3T(row[0].x, row[1].x, row[2].x);
		col[1] = Vec3T(row[0].y, row[1].y, row[2].y);
		col[2] = Vec3T(row[0].z, row[1].z, row[2].z);
		row[0] = col[0];
		row[1] = col[1];
		row[2] = col[2];
	}
};

template <typename T>
Vec3<T> operator*(const Mat3<T>& m, const Vec3<T>& v) {
	Vec3<T> r;
	r.x = m.row[0].dot(v);
	r.y = m.row[1].dot(v);
	r.z = m.row[2].dot(v);
	return r;
}

typedef Mat3<double> Mat3d;
typedef Mat3<float>  Mat3f;

} // namespace gfx
} // namespace imqs