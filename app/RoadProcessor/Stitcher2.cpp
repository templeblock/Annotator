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

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

Error Stitcher2::DoStitch(string videoFile, float zx, float zy, double seconds, int count) {
	video::VideoFile video;
	auto             err = video.OpenFile(videoFile);
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
	flat.Alloc(gfx::ImageFormat::RGBA, frustum.Width, frustum.Height);
	flatPrev.Alloc(gfx::ImageFormat::RGBA, frustum.Width, frustum.Height);
	flat.Fill(0);
	flatPrev.Fill(0);

	auto flatOrigin = CameraToFlat(video.Width(), video.Height(), Vec2f(0, 0), z1, zx, zy);

	Image frame;
	frame.Alloc(gfx::ImageFormat::RGBA, video.Width(), video.Height());

	//err = Rend.Initialize(5120, 4096);
	//err = Rend.Initialize(5120, 6144);
	err = Rend.Initialize(6144, 6144);
	//err = Rend.Initialize(8192, 8192);
	if (!err.OK())
		return err;

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
		int maxInitialVelocity = 100;

		if (i == 0) {
			Vec2f topLeft  = Vec2f((Rend.FBWidth - flat.Width) / 2, Rend.FBHeight - flat.Height);
			Vec2f topRight = topLeft + Vec2f(flat.Width, 0);
			Vec2f botLeft  = topLeft + Vec2f(0, flat.Height);
			Mesh  m(2, 2);
			m.ResetUniformRectangular(topLeft, topRight, botLeft, flat.Width, flat.Height);
			Rend.DrawMesh(m, flat);
			//Image out;
			//Rend.CopyDeviceToImage(Rect32(0, 0, Rend.FBWidth, Rend.FBHeight), 0, 0, out);
			//out.Alloc(ImageFormat::RGBA, 5120, 4096);
			//out.Fill(0xddaa88cc);
			//Rend.CopyDeviceToImage(Rect32(0, 2000, 4120, 3500), 0, 2000, out); // testing glReadPixels
			//out.SavePng("giant.png", true, 1);
			//Rend.SaveToFile("giant.jpeg");
			StitchTopLeft = Vec2f(ceil(Rend.FBWidth / 2 + frustum.X1), Rend.FBHeight - StitchWindowHeight - maxInitialVelocity);
			//StitchVelocity          = Vec2f(0, -StitchWindowHeight - MaxVelocityPx);
			//StitchVelocity          = Vec2f(0, -StitchWindowHeight - MaxVelocityPx);
			PrevBottomMidAlignPoint = botLeft + 0.5f * (topRight - topLeft);
			//StitchTopLeft += StitchVelocity;
		}
		if (i != 0) {
			int  pixelsPerMeshCell  = 60; // intuitively makes sense to have this be more than flow.MatchRadius * 2
			int  pixelsPerAlignCell = flow.MatchRadius * 2;
			Mesh m((flat.Width + pixelsPerAlignCell - 1) / pixelsPerMeshCell, (flat.Height + pixelsPerAlignCell - 1) / pixelsPerMeshCell);
			m.ResetUniformRectangular(Vec2f(0, 0), Vec2f(flat.Width, 0), Vec2f(0, flat.Height), flat.Width, flat.Height);
			//m.SnapToUVPixelEdges(flat.Width, flat.Height);
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
			tsf::print("Alignment delta: %v, %v\n", delta.x, delta.y);
			//StitchVelocity   = bottomMidAlignPoint - PrevBottomMidAlignPoint;
			delta.x = 0; // HACK.. uncertain how to handle this
			// We want the BOTTOM of our alignment window to line up with the BOTTOM of this most recent frame, BUT offset by the
			// current velocity. And AFTER that, add some padding, to account for vehicle velocity changes. But hang on.. the vehicle
			// could slow down OR speed up, so we can't pad it out here. The padding must come from StitchWindowHeight alone.

			// Our goal now, is to predict where the bottom of the next frame will land. This is pretty simple: We take the bottom
			// of the most recent frame, and add the current velocity.
			Vec2f stitchBotLeft = Vec2f(bottomMidAlignPoint.x - StitchWindowWidth / 2, bottomMidAlignPoint.y) + delta;
			// Then, we bring it down by the max incremental V search.
			stitchBotLeft.y += flow.StableVSearchRange;
			float prevLeft    = StitchTopLeft.x;
			StitchTopLeft     = stitchBotLeft - Vec2f(0, StitchWindowHeight);
			StitchTopLeft.x   = prevLeft; // HACK to prevent horizontal drift
			flatToAlignBias.y = -(flat.Height - StitchWindowHeight + flow.StableVSearchRange);

			//StitchTopLeft     = Vec2f(bottomMidAlignPoint.x - StitchWindowWidth / 2, bottomMidAlignPoint.y - StitchWindowHeight) + delta;
			//flatToAlignBias.y = -(flat.Height - StitchWindowHeight + delta.y);

			//Rend.SaveToFile("giant1.jpeg");
			Rend.DrawMesh(m, flat);
			if (i % 4 == 0)
				Rend.SaveToFile("giant2.jpeg");

			PrevBottomMidAlignPoint = bottomMidAlignPoint;
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

int Stitch2(argparse::Args& args) {
	auto      videoFile = args.Params[0];
	float     zx        = atof(args.Params[1].c_str());
	float     zy        = atof(args.Params[2].c_str());
	int       count     = args.GetInt("number");
	double    seek      = atof(args.Get("start").c_str());
	Stitcher2 s;
	auto      err = s.DoStitch(videoFile, zx, zy, seek, count);
	if (!err.OK()) {
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs