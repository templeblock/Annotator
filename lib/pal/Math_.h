// This file has an underscore at the end so that it doesn't alias with the C standard library's math.h
#pragma once

#include <float.h>

namespace imqs {
namespace math {

#define IMQS_PI 3.1415926535897932384626433832795

#define IMQS_DEG2RAD (180 / 3.1415926535897932384626433832795)
#define IMQS_RAD2DEG (3.1415926535897932384626433832795 / 180)

template <typename FT>
class Traits {
public:
};

template <>
class Traits<float> {
public:
	static float Epsilon() { return FLT_EPSILON; }
	static float Min() { return FLT_MIN; }
	static float Max() { return FLT_MAX; }
	static bool  IsNaN(float v) { return v != v; }
	static bool  Finite(float v) { return std::isfinite(v); }
};

template <>
class Traits<double> {
public:
	static double Epsilon() { return DBL_EPSILON; }
	static double Min() { return DBL_MIN; }
	static double Max() { return DBL_MAX; }
	static bool   IsNaN(double v) { return v != v; }
	static bool   Finite(double v) { return std::isfinite(v); }
};

template <>
class Traits<int32_t> {
public:
	static int32_t Epsilon() { return 0; }
	static int32_t Min() { return INT32_MIN; }
	static int32_t Max() { return INT32_MAX; }
	static bool    IsNaN(int32_t v) { return false; }
	static bool    Finite(int32_t v) { return true; }
};

template <>
class Traits<int64_t> {
public:
	static int64_t Epsilon() { return 0; }
	static int64_t Min() { return INT64_MIN; }
	static int64_t Max() { return INT64_MAX; }
	static bool    IsNaN(int64_t v) { return false; }
	static bool    Finite(int64_t v) { return true; }
};

// If v is less than zero, return -1
// If v is greater than zero, return 1
// Otherwise return 0
template <typename TReal>
int SignOrZero(TReal v) {
	if (v < 0)
		return -1;
	if (v > 0)
		return 1;
	return 0;
}

// Compare two values and return -1, 0, +1
// Return -1 if a < b
// Return +1 if a > b
// Return 0 otherwise
template <typename T>
int Compare(const T& a, const T& b) {
	if (a < b)
		return -1;
	if (a > b)
		return 1;
	return 0;
}

template <typename T>
T Clamp(const T& v, const T& vmin, const T& vmax) {
	if (v < vmin)
		return vmin;
	if (v > vmax)
		return vmax;
	return v;
}

template <typename T>
T RoundUpInt(T v, T denom) {
	return (v + denom - 1) & ~(denom - 1);
}

inline bool IsFinite(float v) { return std::isfinite(v); }
inline bool IsFinite(double v) { return std::isfinite(v); }
} // namespace math
} // namespace imqs
