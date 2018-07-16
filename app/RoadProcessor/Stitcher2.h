#pragma once

#include "InfiniteBitmap.h"
#include "MeshRenderer.h"

namespace imqs {
namespace roadproc {

/* Stitcher2 is my second attempt at a big image stitcher.
I am keeping the original Stitcher around, so that we have a fallback that can be used to generate training data,
if this second version ends up taking long to build.
*/
class Stitcher2 {
public:
	gfx::Color8 ClearColor; // In production we'll want this to be black, but during dev it's nice to have it some other color like green

	Stitcher2();

	//int   MaxVelocityPx = 100; // maximum vehicle velocity, from one frame to the next
	Error DoStitch(std::string bitmapDir, std::string videoFile, float zx, float zy, double seconds, int count);

private:
	InfiniteBitmap InfBmp;
	gfx::Rect64    InfBmpView; // Where in InfBmp's world is Rend pointed at
	MeshRenderer   Rend;
	int            StitchWindowWidth  = 0;
	int            StitchWindowHeight = 500;
	gfx::Vec2f     StitchTopLeft;
	gfx::Vec2f     StitchVelocity;
	gfx::Vec2f     PrevBottomMidAlignPoint;

	Error AdjustInfiniteBitmapView(const Mesh& m, gfx::Vec2f travelDirection);
};

} // namespace roadproc
} // namespace imqs