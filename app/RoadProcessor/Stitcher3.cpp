#include "pch.h"
#include "Stitcher3.h"
#include "FeatureTracking.h"
#include "Globals.h"
#include "Perspective.h"
#include "OpticalFlow2.h"
#include "Mesh.h"

// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch3 -n 1 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch3 -n 40 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch3 -n 40 --start 0.7 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch3 -n 30 --start 260 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999

// second video
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch3 --phase 1 -n 30 --start 0 ~/mldata/DSCF3040.MOV ~/DSCF3040-positions.json 0 -0.00095
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch3 --phase 2 -n 200 --start 14 ~/mldata/DSCF3040.MOV ~/DSCF3040-positions.json 0 -0.00095

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

Stitcher3::Stitcher3() {
	ClearColor = Color8(0, 150, 0, 60);
	//ClearColor = Color8(0, 0, 0, 0);
}

Error Stitcher3::Initialize(string bitmapDir, std::vector<std::string> videoFiles, float zx, float zy, double seconds) {
	auto err = InfBmp.Initialize(bitmapDir);
	if (!err.OK())
		return err;

	VidStitcher.EnableFullFlatOutput = true;
	VidStitcher.DebugStartVideoAt    = seconds;
	err                              = VidStitcher.Start(videoFiles, zy);
	if (!err.OK())
		return err;

	//err = Rend.Initialize(4096, 3072);
	//err = Rend.Initialize(5120, 4096);
	//err = Rend.Initialize(5120, 6144);
	//err = Rend.Initialize(6144, 6144);
	//err = Rend.Initialize(7168, 7168);
	//err = Rend.Initialize(7168, 4096);
	err = Rend.Initialize(8192, 8192);
	if (!err.OK())
		return err;
	IMQS_ASSERT(Rend.FBWidth % InfBmp.TileSize == 0);
	IMQS_ASSERT(Rend.FBHeight % InfBmp.TileSize == 0);
	InfBmpView = Rect64(0, 0, Rend.FBWidth, Rend.FBHeight);
	Rend.Clear(ClearColor);

	PrevDir = Vec2f(0, -1);

	return Error();
}

Error Stitcher3::DoStitch(Phases phase, string tempDir, string bitmapDir, std::vector<std::string> videoFiles, std::string trackFile, float zx, float zy, double seconds, int count) {
	TempDir = tempDir;

	string localBitmapDir = bitmapDir;
	if (phase == Phases::InitialStitch) {
		auto meshDir   = path::Join(TempDir, "mesh");
		localBitmapDir = path::Join(TempDir, "inf");
		os::RemoveAll(localBitmapDir);
		os::MkDirAll(localBitmapDir);
		os::RemoveAll(meshDir);
		os::MkDirAll(meshDir);
	} else {
		auto err = Track.LoadFile(trackFile);
		if (!err.OK())
			return err;
		Track.Dump(95, 100, 0.2);
		exit(1);
	}

	auto err = Initialize(localBitmapDir, videoFiles, zx, zy, seconds);
	if (!err.OK())
		return err;

	switch (phase) {
	case Phases::InitialStitch: return DoStitchInitial(count);
	case Phases::GeoReference: return DoGeoReference(count);
	}
	// unreachable
	return Error();
}

Error Stitcher3::DoStitchInitial(int count) {
	for (int i = 0; i < count || count == -1; i++) {
		auto err = VidStitcher.Next();
		if (!err.OK())
			return err;

		err = StitchFrame();
		if (!err.OK())
			return err;

		if (i % 15 == 0 || count < 15)
			Rend.SaveToFile("giant2.jpeg");

		if (VidStitcher.FrameNumber != 0) {
			err = VidStitcher.Mesh.SaveCompact(path::Join(TempDir, "mesh", tsf::fmt("%08d", VidStitcher.FrameNumber)));
			if (!err.OK())
				return err;
		}

		VidStitcher.PrintRemainingTime();

		AdjustInfiniteBitmapView(PrevFullMesh, PrevDir);
	}
	return Error();
}

Error Stitcher3::StitchFrame() {
	// The alignment mesh is produced on a crop of the full flattened frame.
	// Our job now is to turn that little alignment mesh into a full mesh that covers the entire
	// flattened frame.
	// In addition, we need to shift the coordinate frame from the previous flattened frame,
	// to the coordinate frame of our 8192x8192 composite "splat" image.
	auto cropRect = VidStitcher.CropRectFromFullFlat();

	Mesh  full;
	Vec2f uvAtTopLeftOfImage;
	Vec2f debugTopLeft;
	if (VidStitcher.FrameNumber == 0) {
		Vec2f topLeft((Rend.FBWidth - VidStitcher.FullFlat.Width) / 2, Rend.FBHeight - VidStitcher.FullFlat.Height);
		Vec2f topRight = topLeft + Vec2f(VidStitcher.FullFlat.Width, 0);
		Vec2f botLeft  = topLeft + Vec2f(0, VidStitcher.FullFlat.Height);
		full.Initialize(2, 2);
		full.ResetUniformRectangular(topLeft, topRight, botLeft, VidStitcher.FullFlat.Width, VidStitcher.FullFlat.Height);
		uvAtTopLeftOfImage = Vec2f(0, 0);
		debugTopLeft       = topLeft;
	} else {
		ExtrapolateMesh(VidStitcher.Mesh, full, uvAtTopLeftOfImage);
		TransformMeshIntoRendCoords(full);
	}

	Rend.DrawMesh(full, VidStitcher.FullFlat);

	Vec2f newTopLeft = full.PosAtFractionalUV(uvAtTopLeftOfImage);
	Vec2f dir        = newTopLeft - PrevTopLeft;
	if (dir.size() > 5 && VidStitcher.FrameNumber >= 1) {
		// only update direction if we're actually moving
		PrevDir = dir;
	}
	PrevTopLeft  = newTopLeft;
	PrevFullMesh = full;

	return Error();
}

void Stitcher3::ExtrapolateMesh(const Mesh& smallMesh, Mesh& fullMesh, gfx::Vec2f& uvAtTopLeftOfImage) {
	Rect32 smallCropRect = VidStitcher.CropRectFromFullFlat();

	// This is the position of the upper-left corner of the small alignment image, within the full flattened image
	Vec2f smallPosInFullImg(smallCropRect.x1, smallCropRect.y1);

	//Vec2f uvInterval = smallMesh.At(smallMesh.Width - 1, smallMesh.Height - 1).UV - smallMesh.At(0, 0).UV;
	//uvInterval *= Vec2f(1.0f / (float) (smallMesh.Width - 1), 1.0f / (float) (smallMesh.Height - 1));
	Vec2f uvInterval(VidStitcher.PixelsPerMeshCell, VidStitcher.PixelsPerMeshCell);

	// Align to the first valid vertex of smallMesh
	int  alignMX, alignMY;
	bool ok = smallMesh.FirstValid(alignMX, alignMY);
	IMQS_ASSERT(ok);

	// This position of the (somewhat arbitrary) alignment vertex, inside the full flattened image
	Vec2f alignPos = smallMesh.At(alignMX, alignMY).UV + smallPosInFullImg;

	// If we were to compute a uniform full mesh right now, with the origin at (0,0), then
	// what we would be the nearest vertex to alignPos?
	Vec2f correction = Vec2f(fmod(alignPos.x, uvInterval.x), fmod(alignPos.y, uvInterval.y));

	// We shift the origin of our full mesh, by the correction factor, so that we end up with
	// vertices who's UV coordinates precisely match those of the small mesh
	Vec2f fullOrigin = -uvInterval + correction;

	// compute full mesh width/height, now that we know the correction factor
	int fullImgWidth  = VidStitcher.FullFlat.Width;
	int fullImgHeight = VidStitcher.FullFlat.Height;
	int mWidth        = (int) ((fullImgWidth - fullOrigin.x + uvInterval.x - 1) / uvInterval.x);
	int mHeight       = (int) ((fullImgHeight - fullOrigin.y + uvInterval.y - 1) / uvInterval.y);

	// delta in mesh coordinates, from small to full
	int smallToFullDX = 0;
	int smallToFullDY = 0;

	uvAtTopLeftOfImage = -fullOrigin;

	fullMesh.Initialize(mWidth, mHeight);
	for (int y = 0; y < mHeight; y++) {
		for (int x = 0; x < mWidth; x++) {
			Vec2f p = fullOrigin + Vec2f((float) x, (float) y) * uvInterval;
			if (p.distanceSQ(alignPos) < 1.0f) {
				smallToFullDX = x - alignMX;
				smallToFullDY = y - alignMY;
			}
			fullMesh.At(x, y).UV      = p;
			fullMesh.At(x, y).Pos     = p;
			fullMesh.At(x, y).IsValid = false;
		}
	}
	IMQS_ASSERT(smallToFullDX != 0 && smallToFullDY != 0);

	Rect32 validRectFull = Rect32::Inverted();

	// Copy the vertices from the small mesh
	for (int y = 0; y < smallMesh.Height; y++) {
		for (int x = 0; x < smallMesh.Width; x++) {
			if (smallMesh.At(x, y).IsValid) {
				int fx                      = x + smallToFullDX;
				int fy                      = y + smallToFullDY;
				fullMesh.At(fx, fy).IsValid = true;
				// The UV coordinates in the source will have been twiddled so that they fall precisely
				// in between a quad of pixels, so here we bring over that twiddling too. We expect
				// the UV coordinates here to change by at most 1 pixel in X and Y.
				// The Pos coordinates, on the other hand, are the alignment positions, so those we
				// expect to have changed a lot.
				fullMesh.At(fx, fy).UV  = smallMesh.At(x, y).UV + smallPosInFullImg;
				fullMesh.At(fx, fy).Pos = smallMesh.At(x, y).Pos + smallPosInFullImg;
				validRectFull.ExpandToFit(fx, fy);
			}
		}
	}

	// fill in all non-valid values with the average displacement
	Vec2f avgDisp = smallMesh.AvgValidDisplacement();
	for (int i = 0; i < fullMesh.Count; i++) {
		if (!fullMesh.Vertices[i].IsValid) {
			fullMesh.Vertices[i].Pos = fullMesh.Vertices[i].UV + avgDisp;
		}
	}

	/*
	// We create a 'full' mesh that is slightly larger than our flattened image.
	// The vertices of this mesh will not coincide 100% with the vertices of the small mesh,
	// so that is why we make the mesh too big. Then, we move the mesh so that it fits 100%
	// with the small mesh.
	int widthPlus          = VidStitcher.FullFlat.Width + VidStitcher.PixelsPerMeshCell * 2;
	int heightPlus         = VidStitcher.FullFlat.Height + VidStitcher.PixelsPerMeshCell * 2;
	int pixelsPerAlignCell = VidStitcher.Flow.MatchRadius;
	int mWidth             = (widthPlus + pixelsPerAlignCell - 1) / VidStitcher.PixelsPerMeshCell;
	int mHeight            = (heightPlus + pixelsPerAlignCell - 1) / VidStitcher.PixelsPerMeshCell;
	fullMesh.Initialize(mWidth, mHeight);
	fullMesh.ResetIdentityForWarpMesh(widthPlus, heightPlus, VidStitcher.Flow.MatchRadius, false);

	// shift the mesh so that it is centered over the flattened image
	Vec2f offset(VidStitcher.PixelsPerMeshCell, VidStitcher.PixelsPerMeshCell);
	for (int i = 0; i < fullMesh.Count; i++) {
		fullMesh.Vertices[i].Pos -= offset;
		fullMesh.Vertices[i].UV -= offset;
	}

	Rect32 smallCropRect = VidStitcher.CropRectFromFullFlat();

	// This is the position of the upper-left corner of the small alignment image, within the full flattened image
	Vec2f smallPosInFullImg(smallCropRect.x1, smallCropRect.y1);

	// Align to the first valid vertex of smallMesh
	int  alignMX, alignMY;
	bool ok = smallMesh.FirstValid(alignMX, alignMY);
	IMQS_ASSERT(ok);

	Vec2f firstValidPos = smallMesh.At(alignMX, alignMY).UV;
	firstValidPos += smallPosInFullImg; // Bring into the coordinate frame of the full image

	// brute force to find the closest vertex in fullMesh
    float bestDistSQ = FLT_MAX;
	for (int y = 0; y < fullMesh.Height; y++) {
		for (int x = 0; x < fullMesh.Width; x++) {
			float distSQ = fullMesh.At(x, y).UV.distanceSQ(firstValidPos);
            if (dstSQ < bestDistSQ) {
                bestDistSQ
            }
		}
	}
    */
}

void Stitcher3::TransformMeshIntoRendCoords(Mesh& mesh) {
	// thinking.. we want to rotate by current bearing, not by this number here
	// also.. need to do rotation before alignment..
	float angle = -PrevDir.angle(Vec2f(0, -1));
	angle       = 0;
	Vec2f r0(cos(angle), -sin(angle));
	Vec2f r1(sin(angle), cos(angle));
	for (int i = 0; i < mesh.Count; i++) {
		Vec2f p;
		// rotate about the origin
		p.x = mesh.Vertices[i].Pos.dot(r0);
		p.y = mesh.Vertices[i].Pos.dot(r1);
		// translate to the top-left of the previous frame
		p += PrevTopLeft;
		mesh.Vertices[i].Pos = p;
	}
}

Error Stitcher3::DoGeoReference(int count) {
	for (int i = 0; i < count || count == -1; i++) {
	}
	return Error();
}

Error Stitcher3::AdjustInfiniteBitmapView(const Mesh& m, gfx::Vec2f travelDirection) {
	auto isInside = [&](Vec2f p) {
		return p.x >= 0 && p.y >= 0 && p.x < Rend.FBWidth && p.y < Rend.FBHeight;
	};
	// check the two extreme points at the front of the frustum, to detect if we're wandering outside of the framebuffer
	if (isInside(m.At(0, 0).Pos) && isInside(m.At(m.Width - 1, 0).Pos))
		return Error();

	bool persistToInfBmp = Phase == Phases::GeoReference;

	// Persist current framebuffer
	Image img;
	if (persistToInfBmp) {
		Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, img);
		auto err = InfBmp.Save(InfBmpView, img);
		if (!err.OK())
			return err;
	}

	if (false) {
		Image test;
		auto  err = InfBmp.Load(InfBmpView, test);
		test.SaveFile("test-inf-view-1.jpeg");
	}

	// Figure out the best new viewport, given the direction of travel

	// Compute the bounding box of the warp mesh, in InfBmp coordinates
	Rect64  meshBounds = Rect64::Inverted();
	Point32 samples[4] = {{0, 0}, {m.Width - 1, 0}, {m.Width - 1, m.Height - 1}, {0, m.Height - 1}};
	for (int i = 0; i < 4; i++) {
		auto p = m.At(samples[i].x, samples[i].y).Pos;
		meshBounds.ExpandToFit((int64_t) p.x + InfBmpView.x1, (int64_t) p.y + InfBmpView.y1);
	}

	Rect64 newView;
	if (travelDirection.x > 0) {
		newView.x1 = InfiniteBitmap::RoundDown64(meshBounds.x1, InfBmp.TileSize);
		newView.x2 = newView.x1 + Rend.FBWidth;
	} else {
		newView.x2 = InfiniteBitmap::RoundUp64(meshBounds.x2, InfBmp.TileSize);
		newView.x1 = newView.x2 - Rend.FBWidth;
	}

	if (travelDirection.y > 0) {
		newView.y1 = InfiniteBitmap::RoundDown64(meshBounds.y1, InfBmp.TileSize);
		newView.y2 = newView.y1 + Rend.FBHeight;
	} else {
		newView.y2 = InfiniteBitmap::RoundUp64(meshBounds.y2, InfBmp.TileSize);
		newView.y1 = newView.y2 - Rend.FBHeight;
	}

	if (persistToInfBmp) {
		img.Fill(0);
		auto err = InfBmp.Load(newView, img);
		if (!err.OK())
			return err;
		Rend.Clear(ClearColor);
		Rend.CopyImageToDevice(img, 0, 0);
	} else {
		Rend.Clear(ClearColor);
	}

	if (false) {
		img.SaveFile("test-inf-view-load.jpeg");
		Image test;
		Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, test);
		test.SaveFile("test-inf-view-in-FB.jpeg");
	}

	Vec2f adjust = Vec2f(float(InfBmpView.x1 - newView.x1), float(InfBmpView.y1 - newView.y1));
	PrevTopLeft += adjust;
	//StitchTopLeft += adjust;
	//PrevBottomMidAlignPoint += adjust;
	InfBmpView = newView;

	return Error();
}

int WebTiles3(argparse::Args& args) {
	string         bmpDir = args.Params[0];
	InfiniteBitmap bmp;
	auto           err = bmp.Initialize(bmpDir);
	if (err.OK())
		err = bmp.CreateWebTiles();
	if (!err.OK()) {
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

int Stitch3(argparse::Args& args) {
	auto   videoFiles = strings::Split(args.Params[0], ',');
	auto   trackFile  = args.Params[1];
	float  zx         = atof(args.Params[2].c_str());
	float  zy         = atof(args.Params[3].c_str());
	int    count      = args.GetInt("number");
	int    iphase     = args.GetInt("phase");
	double seek       = atof(args.Get("start").c_str());

	Stitcher3::Phases phase;
	if (iphase == 1)
		phase = Stitcher3::Phases::InitialStitch;
	else if (iphase == 2)
		phase = Stitcher3::Phases::GeoReference;

	string    tmpDir = "/home/ben/stitch-temp";
	string    bmpDir = "/home/ben/inf";
	Stitcher3 s;
	auto      err = s.DoStitch(phase, tmpDir, bmpDir, videoFiles, trackFile, zx, zy, seek, count);
	if (!err.OK()) {
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs