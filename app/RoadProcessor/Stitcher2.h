#pragma once

#include "InfiniteBitmap.h"
#include "MeshRenderer.h"
#include "Perspective.h"
#include "PositionTrack.h"

namespace imqs {
namespace roadproc {

/* Stitcher2 is my second attempt at a big image stitcher.
I am keeping the original Stitcher around, so that we have a fallback that can be used to generate training data,
if this second version ends up taking long to build.
*/
class Stitcher2 {
public:
	enum class Phases {
		InitialStitch,
		GeoReference,
	};
	Phases      Phase = Phases::InitialStitch;
	gfx::Color8 ClearColor; // In production we'll want this to be black, but during dev it's nice to have it some other color like green

	Stitcher2();

	Error DoStitch(Phases phase, std::string tempDir, std::string bitmapDir, std::string videoFile, std::string trackFile, float zx, float zy, double seconds, int count);

private:
	std::string      TempDir;
	video::VideoFile Video;
	InfiniteBitmap   InfBmp;
	gfx::Rect64      InfBmpView; // Where in InfBmp's world is Rend pointed at
	MeshRenderer     Rend;
	PositionTrack    Track;
	int              PixelsPerMeshCell  = 60; // it makes sense to have this be more than flow.MatchRadius * 2
	int              StitchWindowWidth  = 0;
	int              StitchWindowHeight = 500;
	gfx::Vec2f       StitchTopLeft;
	gfx::Vec2f       StitchVelocity;
	gfx::Vec2f       PrevBottomMidAlignPoint;
	gfx::Vec2f       CurrentVelocity;

	Error Initialize(std::string bitmapDir, std::string videoFile, float zx, float zy, double seconds, PerspectiveParams& pp, Frustum& frustum, gfx::Vec2f& flatOrigin);
	Error AdjustInfiniteBitmapView(const Mesh& m, gfx::Vec2f travelDirection);
	Error DoStitchInitial(PerspectiveParams pp, Frustum frustum, gfx::Vec2f flatOrigin, double seconds, int count);
	Error DoGeoReference(PerspectiveParams pp, Frustum frustum, gfx::Vec2f flatOrigin, double seconds, int count);
	float AverageBrightness(const gfx::Image& img);
	void  SetupMesh(int srcWidth, int srcHeight, int flowMatchRadius, Mesh& m);
};

} // namespace roadproc
} // namespace imqs