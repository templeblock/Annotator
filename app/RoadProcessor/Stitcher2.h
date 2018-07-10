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
	Error DoStitch(std::string videoFile, float zx, float zy, double seconds, int count);

private:
	InfiniteBitmap InfBmp;
	MeshRenderer   Rend;
	int            StitchWindowWidth  = 0;
	int            StitchWindowHeight = 300;
	gfx::Vec2d     StitchTopLeft;
};

} // namespace roadproc
} // namespace imqs