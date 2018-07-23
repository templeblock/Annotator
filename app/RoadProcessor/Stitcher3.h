#pragma once

#include "InfiniteBitmap.h"
#include "MeshRenderer.h"
#include "Perspective.h"
#include "PositionTrack.h"
#include "VideoStitcher.h"

namespace imqs {
namespace roadproc {

/* Stitcher3 is my third attempt at a big image stitcher.
*/
class Stitcher3 {
public:
	enum class Phases {
		InitialStitch,
		GeoReference,
	};
	Phases      Phase = Phases::InitialStitch;
	gfx::Color8 ClearColor; // In production we'll want this to be black, but during dev it's nice to have it some other color like green

	Stitcher3();

	Error DoStitch(Phases phase, std::string tempDir, std::string bitmapDir, std::vector<std::string> videoFiles, std::string trackFile, float zx, float zy, double seconds, int count);

private:
	VideoStitcher  VidStitcher;
	std::string    TempDir;
	InfiniteBitmap InfBmp;
	gfx::Rect64    InfBmpView; // Where in InfBmp's world is Rend pointed at
	MeshRenderer   Rend;
	PositionTrack  Track;
	gfx::Vec2f     PrevTopLeft;
	gfx::Vec2f     PrevDir; // Direction of the top of the flattened frame. (0, -1) is straight ahead.
	Mesh           PrevFullMesh;

	Error Initialize(std::string bitmapDir, std::vector<std::string> videoFiles, float zx, float zy, double seconds);
	Error AdjustInfiniteBitmapView(const Mesh& m, gfx::Vec2f travelDirection);
	Error DoStitchInitial(int count);
	Error StitchFrame();
	void  ExtrapolateMesh(const Mesh& smallMesh, Mesh& fullMesh, gfx::Vec2f& uvAtTopLeftOfImage);
	void  TransformMeshIntoRendCoords(Mesh& mesh);

	Error DoGeoReference(int count);
};

} // namespace roadproc
} // namespace imqs