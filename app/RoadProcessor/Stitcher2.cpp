#include "pch.h"
#include "Stitcher2.h"
#include "FeatureTracking.h"
#include "Globals.h"
#include "Perspective.h"
#include "OpticalFlow2.h"
#include "Mesh.h"

// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch2 -n 1 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch2 -n 40 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch2 -n 40 --start 0.7 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch2 -n 30 --start 260 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.000999

// TODO: Align the warp mesh so that on the bottom, it has cells which touch right up against the bottom of the image.
// This should help substantially with the alignment in the most visible, highly detailed region - ie the middle of the road.

// Size of 1000 frames (33.3 seconds): 2.7 GB, which is 83 MB/s !!!
// After compressing to JPEG, and 3 levels of tiles, we have 294 MB, which is 8.8 MB/s. How can this be larger than the video??

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

Stitcher2::Stitcher2() {
	//ClearColor = Color8(0, 150, 0, 60);
	ClearColor = Color8(0, 0, 0, 0);
}

Error Stitcher2::DoStitch(string bitmapDir, string videoFile, float zx, float zy, double seconds, int count) {
	auto err = InfBmp.Initialize(bitmapDir);
	if (!err.OK())
		return err;

	video::VideoFile video;
	err = video.OpenFile(videoFile);
	if (!err.OK())
		return err;

	err = video.SeekToSecond(seconds, video::Seek::Any);
	if (!err.OK())
		return err;

	err = global::Lens->InitializeDistortionCorrect(video.Width(), video.Height());
	if (!err.OK())
		return err;

	float z1      = FindZ1ForIdentityScaleAtBottom(video.Width(), video.Height(), zx, zy);
	auto  frustum = ComputeFrustum(video.Width(), video.Height(), z1, zx, zy);
	Image flat, flatPrev;
	flat.Alloc(gfx::ImageFormat::RGBAP, frustum.Width, frustum.Height);
	flatPrev.Alloc(gfx::ImageFormat::RGBAP, frustum.Width, frustum.Height);
	flat.Fill(0);
	flatPrev.Fill(0);

	auto flatOrigin = CameraToFlat(video.Width(), video.Height(), Vec2f(0, 0), z1, zx, zy);

	Image frame;
	frame.Alloc(gfx::ImageFormat::RGBAP, video.Width(), video.Height());

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

	OpticalFlow2 flow;
	// we match a square of 16x16, centered around a pixel crack
	flow.MatchRadius = 8;
	Vec2f flatToAlignBias(0, 0);

	for (int i = 0; i < count; i++) {
		err = video.DecodeFrameRGBA(frame.Width, frame.Height, frame.Data, frame.Stride);
		if (err == ErrEOF)
			break;
		if (!err.OK())
			return err;

		RemovePerspective(frame, flat, z1, zx, zy, (int) flatOrigin.x, (int) flatOrigin.y);

		//flat.SaveJpeg(tsf::fmt("flat-%d.jpeg", i));

		StitchWindowWidth      = floor(frustum.X2 - frustum.X1);
		int maxInitialVelocity = 300;

		if (i == 0) {
			Vec2f topLeft  = Vec2f((Rend.FBWidth - flat.Width) / 2, Rend.FBHeight - flat.Height);
			Vec2f topRight = topLeft + Vec2f(flat.Width, 0);
			Vec2f botLeft  = topLeft + Vec2f(0, flat.Height);
			Mesh  m(2, 2);
			m.ResetUniformRectangular(topLeft, topRight, botLeft, flat.Width, flat.Height);
			Rend.DrawMesh(m, flat);
			//Image out;
			//Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, out);
			//out.Alloc(ImageFormat::RGBAP, 5120, 4096);
			//out.Fill(0xddaa88cc);
			//Rend.CopyDeviceToImage(Rect32(0, 2000, 4120, 3500), 0, 2000, out); // testing glReadPixels
			//out.SavePng("giant.png", true, 1);
			//Rend.SaveToFile("giant.jpeg");
			StitchTopLeft = Vec2f(ceil(Rend.FBWidth / 2 + frustum.X1), Rend.FBHeight - StitchWindowHeight - maxInitialVelocity);
			//StitchVelocity          = Vec2f(0, -StitchWindowHeight - MaxVelocityPx);
			//StitchVelocity          = Vec2f(0, -StitchWindowHeight - MaxVelocityPx);
			PrevBottomMidAlignPoint = botLeft + 0.5f * (topRight - topLeft);
			//StitchTopLeft += StitchVelocity;
			Rend.SaveToFile("giant2.jpeg");
		}
		if (i != 0) {
			int pixelsPerMeshCell  = 60; // it makes sense to have this be more than flow.MatchRadius * 2
			int pixelsPerAlignCell = flow.MatchRadius * 2;
			int mWidth             = (flat.Width + pixelsPerAlignCell - 1) / pixelsPerMeshCell;
			int mHeight            = (flat.Height + pixelsPerAlignCell - 1) / pixelsPerMeshCell;
			if (mWidth % 2 == 0) {
				// Ensure that the grid has an odd number of cells, which guarantees that there is
				// a mesh line running through the horizontal center of the grid (from top to bottom),
				// which is perfectly in the center of the grid. We use this center point to compute
				// horizontal drift, so that's why it's vital that we have it perfectly in the center
				// of the image. See bottomMidAlignPoint and PrevBottomMidAlignPoint.
				mWidth++;
			}
			Mesh m(mWidth, mHeight);
			m.ResetIdentityForWarpMesh(flat.Width, flat.Height, flow.MatchRadius);
			m.SnapToUVPixelEdges();
			//m.Print(Rect32(0, 0, 5, 5), flat.Width, flat.Height);

			// Extract the potential alignment region out of the splat image, and align 'flat' onto it.
			//StitchTopLeft += StitchVelocity;

			// We don't ever adjust flatToAlignBias, unless we drift our stitch alignment point downward, to pick out higher quality pixels (closer to lens)
			//flatToAlignBias += StitchVelocity;

			Image alignTargetImg;
			//int    alignHeight = i == 1 ? StitchWindowHeight + MaxVelocityPx : StitchWindowHeight;
			int    alignHeight = i == 1 ? StitchWindowHeight + maxInitialVelocity : StitchWindowHeight;
			Rect32 alignTargetRect(StitchTopLeft.x, StitchTopLeft.y, StitchTopLeft.x + StitchWindowWidth, StitchTopLeft.y + alignHeight);
			Rend.CopyDeviceToImage(alignTargetRect, 0, 0, alignTargetImg);
			//alignTargetImg.SaveFile("alignTarget.jpeg");

			flow.Frame(m, frustum, flat, alignTargetImg, flatToAlignBias);

			// The mesh target positions are in the coordinate frame of alignTargetImg. Before rendering the mesh,
			// we must convert those coordinates into the coordinate frame of the giant splat image.
			//m.PrintSample(m.Width / 2, m.Height - 1);
			m.TransformTargets(StitchTopLeft);
			//m.PrintSample(m.Width / 2, m.Height / 2);
			//m.PrintSample(m.Width / 2, m.Height - 1);
			Vec2f bottomMidAlignPoint    = m.At(m.Width / 2, m.Height - 1).Pos;
			Vec2f bottomMidAlignPointSrc = m.At(m.Width / 2, m.Height - 1).UV;
			Vec2f delta                  = bottomMidAlignPoint - PrevBottomMidAlignPoint;
			tsf::print("Alignment delta at %.1f: %v, %v\n", video.LastFrameTimeSeconds(), delta.x, delta.y);
			//StitchVelocity   = bottomMidAlignPoint - PrevBottomMidAlignPoint;
			//delta.x = 0; // HACK.. uncertain how to handle this
			// We want the BOTTOM of our alignment window to line up with the BOTTOM of this most recent frame, BUT offset by the
			// current velocity. And AFTER that, add some padding, to account for vehicle velocity changes. But hang on.. the vehicle
			// could slow down OR speed up, so we can't pad it out here. The padding must come from StitchWindowHeight alone.

			// Our goal now, is to predict where the bottom of the next frame will land. This is pretty simple: We take the bottom
			// of the most recent frame, and add the current velocity.
			Vec2f stitchBotLeft = Vec2f(bottomMidAlignPoint.x - StitchWindowWidth / 2, bottomMidAlignPoint.y) + delta;
			// Then, we bring it down by the max incremental V search.
			stitchBotLeft.y += flow.StableVSearchRange;
			float prevLeft  = StitchTopLeft.x;
			StitchTopLeft   = stitchBotLeft - Vec2f(0, StitchWindowHeight);
			StitchTopLeft.x = prevLeft + 0.8 * (StitchTopLeft.x - prevLeft); // necessary hack to minimize the amount of horizontal drift
			//StitchTopLeft.x   = prevLeft; // HACK to prevent horizontal drift
			flatToAlignBias.y = -(flat.Height - StitchWindowHeight + flow.StableVSearchRange);

			//StitchTopLeft     = Vec2f(bottomMidAlignPoint.x - StitchWindowWidth / 2, bottomMidAlignPoint.y - StitchWindowHeight) + delta;
			//flatToAlignBias.y = -(flat.Height - StitchWindowHeight + delta.y);

			//Rend.SaveToFile("giant1.jpeg");
			// Discard the final 3 rows of the warp mesh, so that we avoid using areas of the lens where there's substantial vignetting.
			// We *could* use lensfun to remove the vignetting, but then we need to perform that processing on the GPU, because the color
			// corrections need to happen in linear space. Actually.. now that I write this, I haven't measured the speed of doing it in CPU.
			// It could be acceptable. This constant here is intimately related to the constant in OpticalFlow2.cpp - search for "Rend.DrawMesh" there.
			Rect32 mrect(0, 0, m.Width, m.Height - 3);

			// This is very useful to see where the stitch lines are
			//for (int x = 0; x < m.Width; x++)
			//	m.At(x, m.Height - 4).Color.r = 0;

			Rend.DrawMesh(m, flat, mrect);
			if (i % 8 == 0 || count < 8)
				Rend.SaveToFile("giant2.jpeg");

			PrevBottomMidAlignPoint = bottomMidAlignPoint;

			AdjustInfiniteBitmapView(m, delta);
		}

		if (false) {
			err = flat.SaveFile(tsf::fmt("flat-%04d.jpeg", i));
			if (!err.OK())
				return err;
		}
		//tsf::print("%v/%v\r", i + 1, count);
		fflush(stdout);
		std::swap(flatPrev, flat);
	}
	//Giant.SaveJpeg("giant2.jpeg");
	//tsf::print("\n");
	return Error();
}

Error Stitcher2::AdjustInfiniteBitmapView(const Mesh& m, gfx::Vec2f travelDirection) {
	auto isInside = [&](Vec2f p) {
		return p.x >= 0 && p.y >= 0 && p.x < Rend.FBWidth && p.y < Rend.FBHeight;
	};
	// check the two extreme points at the front of the frustum, to detect if we're wandering outside of the framebuffer
	if (isInside(m.At(0, 0).Pos) && isInside(m.At(m.Width - 1, 0).Pos))
		return Error();

	// Persist current framebuffer
	Image img;
	Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, img);
	auto err = InfBmp.Save(InfBmpView, img);
	if (!err.OK())
		return err;

	if (false) {
		Image test;
		err = InfBmp.Load(InfBmpView, test);
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

	img.Fill(0);
	err = InfBmp.Load(newView, img);
	if (!err.OK())
		return err;
	Rend.Clear(ClearColor);
	Rend.CopyImageToDevice(img, 0, 0);

	if (false) {
		img.SaveFile("test-inf-view-load.jpeg");
		Image test;
		Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, test);
		test.SaveFile("test-inf-view-in-FB.jpeg");
	}

	Vec2f adjust = Vec2f(float(InfBmpView.x1 - newView.x1), float(InfBmpView.y1 - newView.y1));
	StitchTopLeft += adjust;
	PrevBottomMidAlignPoint += adjust;
	InfBmpView = newView;

	return Error();
}

int WebTiles(argparse::Args& args) {
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

int Stitch2(argparse::Args& args) {
	auto   videoFile = args.Params[0];
	float  zx        = atof(args.Params[1].c_str());
	float  zy        = atof(args.Params[2].c_str());
	int    count     = args.GetInt("number");
	double seek      = atof(args.Get("start").c_str());

	string    bmpDir = "/home/ben/inf";
	Stitcher2 s;
	auto      err = s.DoStitch(bmpDir, videoFile, zx, zy, seek, count);
	if (!err.OK()) {
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs