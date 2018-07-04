#include "pch.h"
#include "FeatureTracking.h"
#include "Globals.h"
#include "Perspective.h"
#include "OpticalFlow.h"

// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 2 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV -0.000879688
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 10 --start 5 /home/ben/win/c/mldata/DSCF3023.MOV -0.000879688

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

static void StitchFrames(OpticalFlow& flow, int frameNumber, Frustum f, Image& img1, Image& img2, Image& giant) {
	int windowWidth  = f.X2 - f.X1 - 2; // the -2 is a buffer
	int windowHeight = 800;
	int windowTop    = img1.Height - windowHeight;
	int windowLeft   = (img1.Width - windowWidth) / 2;

	auto img1Crop = img1.Window(windowLeft, windowTop, windowWidth, windowHeight);
	auto img2Crop = img2.Window(windowLeft, windowTop, windowWidth, windowHeight);

	flow.Frame(img1Crop, img2Crop);

	if (false) {
		cv::Mat m1 = ImageToMat(img1Crop);
		cv::Mat m2 = ImageToMat(img2Crop);
		cv::Mat mg1, mg2;
		cv::cvtColor(m1, mg1, cv::COLOR_RGB2GRAY);
		cv::cvtColor(m2, mg2, cv::COLOR_RGB2GRAY);
		int                maxPoints   = 10000;
		double             quality     = 0.01;
		int                minDistance = 5;
		KeyPointSet        kp1, kp2;
		vector<cv::DMatch> matches;
		ComputeKeyPointsAndMatch("FREAK", mg1, mg2, maxPoints, quality, minDistance, false, false, kp1, kp2, matches);
		cv::Mat matchImg;
		cv::drawMatches(m1, kp1.Points, m2, kp2.Points, matches, matchImg);
		auto diag = MatToImage(matchImg);
		diag.SavePng("match.png");
	}
}

static Error DoStitch(string videoFile, float z2, double seconds, int count) {
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

	float z1 = FindZ1ForIdentityScaleAtBottom(video.Width(), video.Height(), z2);
	auto  f  = ComputeFrustum(video.Width(), video.Height(), z1, z2);
	Image flat, flatPrev;
	flat.Alloc(gfx::ImageFormat::RGBA, f.Width, f.Height);
	flatPrev.Alloc(gfx::ImageFormat::RGBA, f.Width, f.Height);
	flat.Fill(0);
	flatPrev.Fill(0);

	auto flatOrigin = CameraToFlat(video.Width(), video.Height(), Vec2f(0, 0), z1, z2);

	Image img;
	img.Alloc(gfx::ImageFormat::RGBA, video.Width(), video.Height());

	Image giant;
	giant.Alloc(gfx::ImageFormat::RGBA, video.Width(), 4000);

	OpticalFlow flow;

	for (int i = 0; i < count; i++) {
		err = video.DecodeFrameRGBA(img.Width, img.Height, img.Data, img.Stride);
		if (err == ErrEOF)
			break;
		if (!err.OK())
			return err;

		RemovePerspective(img, flat, z1, z2, (int) flatOrigin.x, (int) flatOrigin.y);
		if (i != 0)
			StitchFrames(flow, i, f, flatPrev, flat, giant);

		if (false) {
			err = flat.SaveFile(tsf::fmt("flat-%04d.jpeg", i));
			if (!err.OK())
				return err;
		}
		//tsf::print("%v/%v\r", i + 1, count);
		fflush(stdout);
		std::swap(flatPrev, flat);
	}
	//tsf::print("\n");
	return Error();
}

int Stitch(argparse::Args& args) {
	auto   videoFile = args.Params[0];
	float  z2        = atof(args.Params[1].c_str());
	int    count     = args.GetInt("number");
	double seek      = atof(args.Get("start").c_str());
	auto   err       = DoStitch(videoFile, z2, seek, count);
	if (!err.OK()) {
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs