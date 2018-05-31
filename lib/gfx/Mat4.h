#pragma once

#include "Vec4.h"

namespace imqs {
namespace gfx {

template <typename T>
class Mat4 {
public:
	typedef Vec4<T>     Vec4T;
	typedef Vec4Base<T> Vec4BaseT;

	Vec4Base<T> row[4];

	T& m(int _row, int col) { return row[_row][col]; }
	T  m(int _row, int col) const { return row[_row][col]; }

	void MakeIdentity() {
		row[0] = Vec4T(1, 0, 0, 0);
		row[1] = Vec4T(0, 1, 0, 0);
		row[2] = Vec4T(0, 0, 1, 0);
		row[3] = Vec4T(0, 0, 0, 1);
	}

	// Setup a transformation matrix ala glOrtho
	void Ortho(T left, T right, T bottom, T top, T znear, T zfar) {
		T A    = 2 / (right - left);
		T B    = 2 / (top - bottom);
		T C    = -2 / (zfar - znear);
		T tx   = -(right + left) / (right - left);
		T ty   = -(top + bottom) / (top - bottom);
		T tz   = -(zfar + znear) / (zfar - znear);
		row[0] = Vec4T(A, 0, 0, tx);
		row[1] = Vec4T(0, B, 0, ty);
		row[2] = Vec4T(0, 0, C, tz);
		row[3] = Vec4T(0, 0, 0, 1);
	}

	// Natural format is row-major. This will make the matrix column-major (useful for glUniformMatrix4fv on OpenGL ES)
	void Transpose() {
		Vec4T col[4];
		col[0] = Vec4T(row[0].x, row[1].x, row[2].x, row[3].x);
		col[1] = Vec4T(row[0].y, row[1].y, row[2].y, row[3].y);
		col[2] = Vec4T(row[0].z, row[1].z, row[2].z, row[3].z);
		col[3] = Vec4T(row[0].w, row[1].w, row[2].w, row[3].w);
		row[0] = col[0];
		row[1] = col[1];
		row[2] = col[2];
		row[3] = col[3];
	}
};

typedef Mat4<double> Mat4d;
typedef Mat4<float>  Mat4f;

} // namespace gfx
} // namespace imqs