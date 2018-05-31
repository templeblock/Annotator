#pragma once

#include <cmath>

namespace imqs {
namespace gfx {

// Base has no default constructor, so it can be included inside unions
template <typename T>
class Vec4Base {
public:
	static const size_t Dim = 4;
	typedef T           FT;
	union {
		struct {
			T x, y, z, w;
		};
		T n[4];
	};

	T  operator[](int col) const { return n[col]; }
	T& operator[](int col) { return n[col]; }

	T dot(Vec4Base b) const {
		return x * b.x + y * b.y + z * b.z + w * b.w;
	}
};

template <typename T>
Vec4Base<T> operator*(Vec4Base<T> v, T s) { return {v.x * s, v.y * s, v.z * s, v.w * s}; }

template <typename T>
Vec4Base<T> operator*(T s, Vec4Base<T> v) { return {v.x * s, v.y * s, v.z * s, v.w * s}; }

template <typename T>
Vec4Base<T> operator/(Vec4Base<T> v, T s) { return {v.x / s, v.y / s, v.z / s, v.w / s}; }

template <typename T>
Vec4Base<T> operator-(Vec4Base<T> a, Vec4Base<T> b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }

template <typename T>
Vec4Base<T> operator+(Vec4Base<T> a, Vec4Base<T> b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }

template <typename T>
class Vec4 : public Vec4Base<T> {
public:
	Vec4() {}
	// Vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {} - MSVC 2015 doesn't like this, but I don't understand why. It says "x is not a base or member" and likewise for y,z,w.
	Vec4(T _x, T _y, T _z, T _w) {
		this->x = _x;
		this->y = _y;
		this->z = _z;
		this->w = _w;
	}
	Vec4(Vec4Base<T> b) {
		this->x = b.x;
		this->y = b.y;
		this->z = b.z;
		this->w = b.w;
	}
};

typedef Vec4<double> Vec4d;
typedef Vec4<float>  Vec4f;

} // namespace gfx
} // namespace imqs