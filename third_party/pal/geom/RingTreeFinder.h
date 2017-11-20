#pragma once

#include "BBox2.h"

namespace imqs {
namespace geom2d {

// RingTreeFinder takes as input, one or more polygons, which we call rings.
// When you run Analyze(), it figures out which polygon is inside which
// other polygon. The end result is a tree of polygons, where children
// are spatially inside their parents. This assumes that polygons do
// not intersect each other, so it only tests the first point of a polygon
// for inclusion inside other polygons.
// This was built for fixing the orientation of polygons (clockwise vs counter-clockwise)
// for data serialization formats such as Shapefile or Well Known Binary.
class IMQS_PAL_API RingTreeFinder {
public:
	struct Point {
		double X, Y;
	};

	struct Ring {
		size_t             Index  = -1;
		int                Level  = -1; // 0 = outer most
		double             Area   = 0;
		Ring*              Parent = nullptr;
		BBox2d             Bounds;
		std::vector<Point> Vertices;
		std::vector<Ring*> Children;
	};

	std::vector<Ring*> Rings;

	RingTreeFinder();
	~RingTreeFinder();

	void AddRing(size_t n, const double* v, int strideInDoubles);
	void AddRing(size_t n, const float* v, int strideInFloats);
	void Analyze(); // Result is in "Rings"

protected:
	template <typename T>
	void AddRingInternal(size_t n, const T* vertices, int strideInT);
	void WalkSetDepth(Ring* r, int level);
};

} // namespace geom2d
} // namespace imqs