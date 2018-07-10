#include "pch.h"
#include "Stitcher2.h"
#include "FeatureTracking.h"
#include "Globals.h"
#include "Perspective.h"
#include "OpticalFlow.h"
#include "Mesh.h"

// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch2 -n 1 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.0009
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch2 -n 5 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV 0 -0.0009

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

	float z1 = FindZ1ForIdentityScaleAtBottom(video.Width(), video.Height(), zx, zy);
	auto  f  = ComputeFrustum(video.Width(), video.Height(), z1, zx, zy);
	Image flat, flatPrev;
	flat.Alloc(gfx::ImageFormat::RGBA, f.Width, f.Height);
	flatPrev.Alloc(gfx::ImageFormat::RGBA, f.Width, f.Height);
	flat.Fill(0);
	flatPrev.Fill(0);

	auto flatOrigin = CameraToFlat(video.Width(), video.Height(), Vec2f(0, 0), z1, zx, zy);

	Image frame;
	frame.Alloc(gfx::ImageFormat::RGBA, video.Width(), video.Height());

	err = Rend.Initialize(5120, 4096);
	//err = Rend.Initialize(8192, 8192);
	if (!err.OK())
		return err;

	OpticalFlow flow;

	for (int i = 0; i < count; i++) {
		err = video.DecodeFrameRGBA(frame.Width, frame.Height, frame.Data, frame.Stride);
		if (err == ErrEOF)
			break;
		if (!err.OK())
			return err;

		RemovePerspective(frame, flat, z1, zx, zy, (int) flatOrigin.x, (int) flatOrigin.y);

		flat.SaveJpeg(tsf::fmt("flat-%d.jpeg", i));

		if (i == 0) {
			Vec2f topLeft  = Vec2f((Rend.FBWidth - flat.Width) / 2, Rend.FBHeight - flat.Height);
			Vec2f topRight = topLeft + Vec2f(flat.Width, 0);
			Vec2f botLeft  = topLeft + Vec2f(0, flat.Height);
			Mesh  m(2, 2);
			m.ResetUniformRectangular(topLeft, topRight, botLeft);
			Rend.DrawMesh(m, flat);
			Image out;
			Rend.CopyDeviceToImage(out);
			out.SavePng("giant.png", true, 1);
		}
		if (i != 0) {
			int ax = (int) StitchTopLeft.x;
			int ay = (int) StitchTopLeft.y;
			//auto alignWindowSrc   = Giant.Window(ax, ay, StitchWindowWidth, StitchWindowHeight);
			int  dstLeft          = (flat.Width - StitchWindowWidth) / 2;
			auto alignWindowDst   = flat.Window(dstLeft, flat.Height - StitchWindowHeight * 3, StitchWindowWidth, StitchWindowHeight * 2.6);
			auto alignWindowBlend = flat.Window(dstLeft, flat.Height - StitchWindowHeight * 3, StitchWindowWidth + 10, StitchWindowHeight * 2.8);
			flow.FirstFrameBiasH  = 0;
			flow.FirstFrameBiasV  = StitchWindowHeight;
			//StitchFrames(flow, i, f, alignWindow, flat);
			//StitchFrames(flow, i, f, flatPrev, flat, giant);
			// alignWindow will have to be rotated by StitchAngle in future... once we support rotation
			//flow.Frame(alignWindowSrc, alignWindowDst);
			//Giant.SaveJpeg("giant1.jpeg");
			//BlendNewFrame(flow, alignWindowBlend);
			//Giant.SaveJpeg("giant2.jpeg");
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