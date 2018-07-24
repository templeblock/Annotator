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
	gfx::Color8 ClearColor;         // In production we'll want this to be black, but during dev it's nice to have it some other color like green
	float       MetersPerPixel = 0; // Computed scale. After computing this, we assume it is constant for the entire recording
	int         BaseZoomLevel  = 0;

	Stitcher3();

	Error DoStitch(std::string bitmapDir, std::vector<std::string> videoFiles, std::string trackFile, float zx, float zy, double seconds, int count);

private:
	struct FrameObject {
		gfx::Vec2f     AvgDisp;
		roadproc::Mesh Mesh;
		gfx::Image     Img;
		double         FrameTime = 0;
	};
	VideoStitcher            VidStitcher;
	InfiniteBitmap           InfBmp;
	gfx::Rect64              InfBmpView; // Where in InfBmp's world is Rend pointed at
	MeshRenderer             Rend;
	PositionTrack            Track;
	gfx::Vec2f               PrevTopLeft;
	gfx::Vec2f               PrevDir;           // Direction of the top of the flattened frame. (0, -1) is straight ahead. Frozen when vehicle is stopped.
	gfx::Vec2f               PrevDirUnfiltered; // Same as PrevDir, but this value is written to regardless of current velocity.
	Mesh                     PrevFullMesh;
	std::vector<FrameObject> Frames;
	bool                     EnableSimpleRender = false;
	bool                     EnableGeoRender    = false;
	bool                     HavePositions      = false;

	Error  Initialize(std::string bitmapDir, std::vector<std::string> videoFiles, float zx, float zy, double seconds);
	Error  AdjustInfiniteBitmapView(const Mesh& m, gfx::Vec2f travelDirection);
	Error  AdjustInfiniteBitmapViewForGeo(gfx::Rect64 outRect);
	Error  MeasurePixelScale();
	void   SetupBaseMapScale();
	double BaseMapMetersPerPixel();
	Error  Run(int count);
	Error  StitchFrame();
	Error  DrawGeoReferencedFrame();
	void   TransformFrameCoordsToGeo(gfx::Vec3d& geoOffset);
	Error  DrawGeoMesh(gfx::Vec3d geoOffset);
	void   ExtrapolateMesh(const Mesh& smallMesh, Mesh& fullMesh, gfx::Vec2f& uvAtTopLeftOfImage);
	void   TransformMeshIntoRendCoords(Mesh& mesh);
};

} // namespace roadproc
} // namespace imqs