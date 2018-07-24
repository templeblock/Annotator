#include "pch.h"
#include "VideoStitcher.h"
#include "Perspective.h"
#include "Globals.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

const double RAD2DEG = 180.0 / IMQS_PI;

Error VideoStitcher::Start(std::vector<std::string> videoFiles, float perspectiveZY) {
	// Before we start this potentially lengthy process, make sure we can open every one of the video files specified.
	// Also, count their total time, and extract the creation time of the first video
	VideoWidth             = 0;
	VideoHeight            = 0;
	TotalVideoSeconds      = 0;
	FirstVideoCreationTime = time::Time();
	VideoFiles             = videoFiles;
	for (size_t i = 0; i < videoFiles.size(); i++) {
		const auto&      v = videoFiles[i];
		video::VideoFile video;
		auto             err = video.OpenFile(v);
		if (!err.OK())
			return err;
		if (i == 0) {
			VideoWidth             = video.Width();
			VideoHeight            = video.Height();
			FirstVideoCreationTime = video.Metadata().CreationTime;
			// Assume camera time is in SAST (UTC+2). Obviously we'll need a different system going forward,
			// for customers not in SAST
			FirstVideoCreationTime -= 2 * time::Hour;
		} else {
			if (video.Width() != VideoWidth || video.Height() != VideoHeight)
				return Error::Fmt("All video files must be the same resolution (%v x %v), (%v x %v)", VideoWidth, VideoHeight, video.Width(), video.Height());
		}
		TotalVideoSeconds += video.GetVideoStreamInfo().DurationSeconds();
	}

	if (FirstVideoCreationTime.IsNull())
		return Error("Unable to extraction creation time metadata from video file");

	if (global::Lens != nullptr) {
		auto err = global::Lens->InitializeDistortionCorrect(VideoWidth, VideoHeight);
		if (!err.OK())
			return err;
	}

	PP.ZX   = 0;
	PP.ZY   = perspectiveZY;
	PP.Z1   = FindZ1ForIdentityScaleAtBottom(VideoWidth, VideoHeight, PP.ZX, PP.ZY);
	Frustum = ComputeFrustum(VideoWidth, VideoHeight, PP);

	// We only monitor a cropped region of the image:
	// +----------+
	// |          |
	// |   +--+   |
	// |   |xx|   |
	// +---+--+---+

	// Splat is only necessary for CPU perspective removal
	if (EnableCPUPerspectiveRemoval)
		Splat.Alloc(ImageFormat::RGBA, Frustum.Width, Frustum.Height);

	if (EnableFullFlatOutput)
		FullFlat.Alloc(ImageFormat::RGBA, Frustum.Width, Frustum.Height);

	// for speed computation
	FlatWidth = Frustum.X2 - Frustum.X1 - 2;

	Flat.Alloc(ImageFormat::RGBA, FlatWidth, FlatHeight);
	FlatPrev.Alloc(ImageFormat::RGBA, FlatWidth, FlatHeight);

	//auto flatOrigin = CameraToFlat(VideoWidth, VideoHeight, Vec2f(0, 0), PP);

	Frame.Alloc(ImageFormat::RGBA, VideoWidth, VideoHeight);

	if (!EnableCPUPerspectiveRemoval) {
		auto err = Rend.Initialize(Frustum.Width, Frustum.Height);
		if (!err.OK())
			return err;
	}

	Flow.StableHSearchRange = 10;
	Flow.StableVSearchRange = 10;

	auto err = Rewind();
	if (!err.OK())
		return err;

	return Error();
}

Error VideoStitcher::Rewind() {
	CurrentVideo  = 0;
	FrameNumber   = -1;
	RemainingTime = time::Duration(0);
	Velocities.clear();

	auto err = Video.OpenFile(VideoFiles[0]);
	if (!err.OK())
		return err;

	if (DebugStartVideoAt != 0) {
		err = Video.SeekToSecond(DebugStartVideoAt, video::Seek::Any);
		if (!err.OK())
			return err;
	}

	// reset tracking parameters
	FlowBias = Vec2f(0, 0);
	AvgDir   = Vec2f(0, -1);
	for (int i = 0; i < 5; i++)
		AbsFlowBias.push_back(Vec2f(0, 0));
	AbsRestart = AbsRestartCheckInterval;
	NeedResync = false;

	return Error();
}

Error VideoStitcher::Next() {
	auto err = LoadNextFrame();
	if (!err.OK())
		return err;

	if (FrameNumber == 0) {
		ProcessingStartTime = time::Now();
		Velocities.emplace_back(FrameTime, Vec2f(0, 0)); // velocity will get adjusted when frame 1 is processed
	} else {
		ComputeTimeRemaining();
		ComputeStitch();
	}

	std::swap(Flat, FlatPrev);

	return Error();
}

Rect32 VideoStitcher::CropRectFromFullFlat() {
	Rect32 crop((Frustum.Width - FlatWidth) / 2, Frustum.Height - FlatHeight, 0, 0);
	crop.x2 = crop.x1 + FlatWidth;
	crop.y2 = crop.y1 + FlatHeight;
	return crop;
}

Error VideoStitcher::LoadNextFrame() {
	auto err = Video.DecodeFrameRGBA(Frame.Width, Frame.Height, Frame.Data, Frame.Stride);
	if (err == ErrEOF) {
		if (CurrentVideo == VideoFiles.size() - 1) {
			// end of the end
			return ErrEOF;
		}

		// You might be tempted to add one frame worth of delay here to VideoTimeOffset, but empirical measurements on our
		// Fuji X-T2 show that this formulation here is correct.
		VideoTimeOffset += Video.LastFrameTimeSeconds();

		CurrentVideo++;
		err = Video.OpenFile(VideoFiles[CurrentVideo]);
		if (!err.OK())
			return err;
		err = Video.DecodeFrameRGBA(Frame.Width, Frame.Height, Frame.Data, Frame.Stride);
		if (!err.OK())
			return err;
	} else if (!err.OK()) {
		return err;
	}
	FrameNumber++;
	FrameTime = VideoTimeOffset + Video.LastFrameTimeSeconds();

	RemovePerspective();

	return Error();
}

void VideoStitcher::ComputeTimeRemaining() {
	double relProcessingSpeed = FrameTime / (time::Now() - ProcessingStartTime).Seconds();
	double remain             = (TotalVideoSeconds - FrameTime) / relProcessingSpeed;
	RemainingTime             = (int64_t)(remain * 1000) * time::Millisecond;
}

void VideoStitcher::RemovePerspective() {
	// Benchmarks
	// CPU:							6:56 minutes
	// GPU without copyback to CPU: 2:56
	// GPU with copyback to CPU:	6:00     -- the culprit is glReadPixels/GPU latency.
	// Only decode video:			2:40

	Rect32 cropRect = CropRectFromFullFlat();

	if (EnableCPUPerspectiveRemoval) {
		// CPU
		auto flatOrigin = CameraToFlat(VideoWidth, VideoHeight, Vec2f(0, 0), PP);
		//roadproc::RemovePerspective(Frame, Flat, PP, flatOrigin.x, flatOrigin.y);
		roadproc::RemovePerspective(Frame, Splat, PP, flatOrigin.x, flatOrigin.y);
		Flat.CopyFrom(Splat, cropRect, 0, 0);
		//Flat.SaveJpeg("speed2-flat-CPU.jpeg");
		if (EnableFullFlatOutput) {
			// This is a wasteful copy. But I don't see us using the CPU path in production
			FullFlat = Splat;
		}
	} else {
		// GPU:
		Rend.Clear(Color8(0, 0, 0, 0));
		Rend.RemovePerspective(Frame, PP);
		//rend.CopyDeviceToImage(Rect32(0, 0, frustum.Width, frustum.Height), 0, 0, flat);
		//flat.SaveJpeg("speed2-flat-GPU.jpeg");
		//exit(1);
		if (EnableFullFlatOutput) {
			Rend.CopyDeviceToImage(Rect32(0, 0, FullFlat.Width, FullFlat.Height), 0, 0, FullFlat);
			Flat.CopyFrom(FullFlat, cropRect, 0, 0);
		} else {
			Rend.CopyDeviceToImage(cropRect, 0, 0, Flat);
		}
	}

	//flat.SaveFile(tsf::fmt("flat-%d.png", iFrame));
	//if (iFrame == 2)
	//	exit(1);
}

Error VideoStitcher::ComputeStitch() {
	FlowResult absFlowResult;
	bool       didReset = false;
	CheckSyncRestart(absFlowResult, didReset);

	SetupMesh(Mesh);
	auto flowResult = Flow.Frame(Mesh, roadproc::Frustum(), Flat, FlatPrev, FlowBias);
	auto disp       = Mesh.AvgValidDisplacement();
	if (disp.size() == 0) {
		// do this to avoid any infinities
		disp = Vec2f(0, -0.01f);
	}
	bool  isStopped = disp.size() < 3.0f;
	float angle     = RAD2DEG * atan2(disp.y, disp.x);
	float angleNorm = RAD2DEG * atan2(AvgDir.y, AvgDir.x);
	if (FrameNumber == 1) {
		if (fabs(angle - AngleDeadAhead) < ExpectedAngleRange)
			AvgDir = disp.normalized();
		FlowBias   = disp;
		AvgImgDiff = flowResult.Diff;
	} else {
		float angleFromExpected = RAD2DEG * disp.angle(AvgDir);
		if (fabs(angleFromExpected) > 10.0f && !isStopped) {
			// make sure we get a restart check soon, if our angle deviates substantially from the norm
			AbsRestart = min(AbsRestart, 15);
		}

		if (fabs(angle - AngleDeadAhead) < ExpectedAngleRange && !isStopped) {
			AvgDir += 0.005f * (disp.normalized() - AvgDir);
		}
		// Bad locks that I've seen go from a diff of 2426.7 to 17710.6. Wondering if that's not an overflow bug...... yeah.. probably overflow.
		if (flowResult.Diff < AvgImgDiff * 2) {
			FlowBias += 0.15f * (disp - FlowBias);
			AvgImgDiff += 0.05f * (flowResult.Diff - AvgImgDiff);
		}
	}
	if (EnableDebugPrint)
		tsf::print("%4.1f: %6.1f, %6.1f (%6.1f, %5.1f) (%6.1f, %6.1f) [%7.1f, %7.1f, %7.1f] %v\n", FrameTime, disp.x, disp.y, angle, angleNorm, FlowBias.x, FlowBias.y, flowResult.Diff, AvgImgDiff, absFlowResult.Diff, didReset ? "RESET" : "");

	Velocities.emplace_back(FrameTime, disp);
	if (FrameNumber == 1) {
		// frame 0 has no history, so we just make it identical to frame 1
		Velocities[0].second = disp;
	}

	return Error();
}

void VideoStitcher::CheckSyncRestart(FlowResult& absFlowResult, bool& didReset) {
	float absErr = 0;
	//Vec2f absDisp(0, 0);
	absFlowResult = FlowResult();
	didReset      = false;
	if (AbsRestart <= 0 || NeedResync) {
		roadproc::Mesh mesh;
		SetupMesh(mesh);
		Vec2f        bias(0, 0);
		OpticalFlow2 flowAbs;
		absFlowResult       = flowAbs.Frame(mesh, roadproc::Frustum(), Flat, FlatPrev, bias);
		auto  disp          = mesh.AvgValidDisplacement();
		float absDivergence = (disp - FlowBias).size();
		if (!NeedResync && absDivergence > AbsDivergenceThreshold && absFlowResult.Diff < AvgImgDiff) {
			NeedResync = true;
		}
		if (NeedResync) {
			bool good = absFlowResult.Diff < AvgImgDiff;
			for (auto d : AbsFlowBias) {
				if (d.distance(disp) > 20)
					good = false;
			}
			if (good) {
				// resync
				//tsf::print("Reset absolute (%v, %v)\n", disp.x, disp.y);
				didReset   = true;
				NeedResync = false;
				FlowBias   = disp;
			}
		}
		AbsFlowBias.erase(AbsFlowBias.begin());
		AbsFlowBias.push_back(disp);
		//absDisp    = disp;
		AbsRestart = AbsRestartCheckInterval;
	} else {
		AbsRestart--;
	}
}

void VideoStitcher::PrintRemainingTime() {
	int min = (int) RemainingTime.Minutes();
	int sec = (int) (RemainingTime.Seconds() - min * 60);
	tsf::print("\rTime remaining: %v:%02d", min, sec);
	fflush(stdout);
}

void VideoStitcher::SetupMesh(roadproc::Mesh& m) {
	SetupMesh(Flat.Width, Flat.Height, MatchHeight, PixelsPerMeshCell, Flow.MatchRadius, m);
}

void VideoStitcher::SetupMesh(int srcWidth, int srcHeight, int matchHeight, int pixelsPerMeshCell, int flowMatchRadius, roadproc::Mesh& m) {
	// matchHeight is something like 200, and srcHeight is something like 500
	// We are only interested in matching the bottom 200 pixels, to the "previous" frame, which
	// is something like 500 pixels big (all frames are the same size).
	// So we generate a mesh for an image that is 200 pixels high, and then we shift
	// that mesh down so that it is aligned to the bottom of the image.
	int pixelsPerAlignCell = flowMatchRadius;
	int mWidth             = (srcWidth + pixelsPerAlignCell - 1) / pixelsPerMeshCell;
	int mHeight            = (matchHeight + pixelsPerAlignCell - 1) / pixelsPerMeshCell;
	m.Initialize(mWidth, mHeight);
	m.ResetIdentityForWarpMesh(srcWidth, matchHeight, flowMatchRadius, false);
	// move the mesh to the bottom of the image
	for (int i = 0; i < m.Count; i++) {
		m.Vertices[i].Pos.y += srcHeight - matchHeight;
		m.Vertices[i].UV.y += srcHeight - matchHeight;
	}
	m.SnapToUVPixelEdges();
}

} // namespace roadproc
} // namespace imqs