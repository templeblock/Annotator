#include "pch.h"
#include "Globals.h"
#include "Perspective.h"
#include "MeshRenderer.h"
#include "OpticalFlow2.h"

// This is the second version of our vehicle speed computation system, which uses
// optical flow on flattened images, instead of using OpenCV feature tracking.

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

enum class SpeedOutputMode {
	CSV,
	JSON,
};

const double RAD2DEG = 180.0 / IMQS_PI;

static void SetupMesh(int srcWidth, int srcHeight, int matchHeight, int pixelsPerMeshCell, int flowMatchRadius, Mesh& m) {
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

static Error DoSpeed2(vector<string> videoFiles, SpeedOutputMode outputMode, string outputFile) {
	FILE* outf = stdout;
	if (outputFile != "stdout") {
		outf = fopen(outputFile.c_str(), "w");
		if (!outf)
			return Error::Fmt("Failed to open output file '%v'", outputFile);
	}

	if (outputMode == SpeedOutputMode::CSV)
		tsf::print(outf, "time,speed\n");

	int videoWidth  = 0;
	int videoHeight = 0;

	// Before we start this potentially lengthy process, make sure we can open every one of the video files specified.
	// Also, count their total time, and extract the creation time of the first video
	double     totalVideoSeconds = 0;
	time::Time firstVideoCreationTime;
	for (size_t i = 0; i < videoFiles.size(); i++) {
		const auto&      v = videoFiles[i];
		video::VideoFile video;
		auto             err = video.OpenFile(v);
		if (!err.OK())
			return err;
		if (i == 0) {
			videoWidth             = video.Width();
			videoHeight            = video.Height();
			firstVideoCreationTime = video.Metadata().CreationTime;
			// Assume camera time is in SAST (UTC+2). Obviously we'll need a different system going forward,
			// for customers not in SAST
			firstVideoCreationTime -= 2 * time::Hour;
			if (outputMode == SpeedOutputMode::JSON)
				tsf::print("Video creation time: %v\n", firstVideoCreationTime.Format8601());
		}
		totalVideoSeconds += video.GetVideoStreamInfo().DurationSeconds();
	}

	if (firstVideoCreationTime.IsNull())
		return Error("Unable to extraction creation time metadata from video file");

	if (global::Lens != nullptr) {
		auto err = global::Lens->InitializeDistortionCorrect(videoWidth, videoHeight);
		if (!err.OK())
			return err;
	}

	PerspectiveParams pp;
	pp.ZX        = 0;
	pp.ZY        = -0.00095;
	pp.Z1        = FindZ1ForIdentityScaleAtBottom(videoWidth, videoHeight, pp.ZX, pp.ZY);
	auto frustum = ComputeFrustum(videoWidth, videoHeight, pp);

	// We only monitor a cropped region of the image:
	// +----------+
	// |          |
	// |   +--+   |
	// |   |xx|   |
	// +---+--+---+
	// That region (indicated by xx) has the best speed-related movement inside it
	Image flat, flatPrev;

	// for testing perspective removal
	int flatWidth  = frustum.Width;
	int flatHeight = frustum.Height;

	// only necessary for CPU perspective removal
	Image splat;
	splat.Alloc(ImageFormat::RGBA, flatWidth, flatHeight);

	// for speed computation
	flatWidth             = frustum.X2 - frustum.X1 - 2;
	flatHeight            = 550;
	int matchHeight       = 150; // only perform matching on the bottom 200 pixels
	int pixelsPerMeshCell = 60;

	flat.Alloc(ImageFormat::RGBA, flatWidth, flatHeight);
	flatPrev.Alloc(ImageFormat::RGBA, flatWidth, flatHeight);

	//auto flatOrigin = CameraToFlat(videoWidth, videoHeight, Vec2f(0, videoHeight * 3 / 8), pp);
	auto flatOrigin = CameraToFlat(videoWidth, videoHeight, Vec2f(0, 0), pp);

	// speeds for every frame except for frame 0, as [time,speed]
	vector<pair<double, double>> speeds;

	auto   startTime       = time::Now();
	double videoTimeOffset = 0; // accumulating time counter, so that we can merge multiple videos into one timelime

	Error err;

	MeshRenderer rend;
	err = rend.Initialize(frustum.Width, frustum.Height);
	if (!err.OK())
		return err;

	OpticalFlow2 flow;
	flow.StableHSearchRange = 10;
	flow.StableVSearchRange = 10;

	for (size_t ivideo = 0; ivideo < videoFiles.size(); ivideo++) {
		video::VideoFile video;
		err = video.OpenFile(videoFiles[ivideo]);
		if (!err.OK())
			return err;

		// HACK
		// 50 is easy, and 50.5
		// 152 is tricky
		// 215 is very tricky
		//video.SeekToSecond(150);
		//video.SeekToSecond(215);
		//video.SeekToSecond(220);
		//video.SeekToSecond(49);

		float  expectedAngleRange = 7; // expect 7 degrees left or right, of "crab walk".. ie camera not facing forwards
		int    iFrame             = 0;
		double frameTime          = 0;
		float  avgImgDiff         = 0;
		Image  frame(ImageFormat::RGBA, videoWidth, videoHeight);
		Vec2f  flowBias(0, 0);
		Vec2f  avgDir(0, 0);
		enum class syncModes {
			fine,
			alert,
			resync,
		} syncMode = syncModes::fine;
		//Vec2f         flowBiasAbsRecover(0, 0);
		vector<Vec2f> flowBiasAbs;
		for (int i = 0; i < 5; i++)
			flowBiasAbs.push_back(Vec2f(0, 0));
		const int absRestartCheckInterval = 100;
		int       absRestart              = absRestartCheckInterval;

		while (err.OK()) {
			err = video.DecodeFrameRGBA(frame.Width, frame.Height, frame.Data, frame.Stride);
			if (err == ErrEOF) {
				err = Error();
				break;
			} else if (!err.OK()) {
				break;
			}
			frameTime = video.LastFrameTimeSeconds();

			// Benchmarks
			// CPU:							6:56 minutes
			// GPU without copyback to CPU: 2:56
			// GPU with copyback to CPU:	6:00     -- the culprit is glReadPixels/GPU latency.
			// Only decode video:			2:40

			Rect32 cropRect((frustum.Width - flatWidth) / 2, frustum.Height - flatHeight, 0, 0);
			cropRect.x2 = cropRect.x1 + flatWidth;
			cropRect.y2 = cropRect.y1 + flatHeight;

			// CPU
			//RemovePerspective(frame, flat, pp, flatOrigin.x, flatOrigin.y);
			//RemovePerspective(frame, splat, pp, flatOrigin.x, flatOrigin.y);
			//flat.CopyFrom(splat, cropRect, 0, 0);
			//flat.SaveJpeg("speed2-flat-CPU.jpeg");

			// GPU:
			rend.Clear(Color8(0, 0, 0, 0));
			rend.RemovePerspective(frame, pp);
			//rend.CopyDeviceToImage(Rect32(0, 0, frustum.Width, frustum.Height), 0, 0, flat);
			//flat.SaveJpeg("speed2-flat-GPU.jpeg");
			rend.CopyDeviceToImage(cropRect, 0, 0, flat);
			//exit(1);

			//flat.SaveFile(tsf::fmt("flat-%d.png", iFrame));
			//if (iFrame == 2)
			//	exit(1);

			if (iFrame >= 1) {
				float      absErr = 0;
				Vec2f      absDisp(0, 0);
				FlowResult absFlowResult;
				bool       didReset = false;
				if (absRestart <= 0 || syncMode != syncModes::fine) {
					// Every 100 frames, perform an absolute lock, and see if it differs from our moving estimate
					Mesh mesh;
					SetupMesh(flat.Width, flat.Height, matchHeight, pixelsPerMeshCell, flow.MatchRadius, mesh);
					Vec2f        bias(0, 0);
					OpticalFlow2 flowAbs;
					absFlowResult       = flowAbs.Frame(mesh, Frustum(), flat, flatPrev, bias);
					auto  disp          = mesh.AvgValidDisplacement();
					float absDivergence = (disp - flowBias).size();
					if (absDivergence > 30 && syncMode == syncModes::fine && absFlowResult.Diff < avgImgDiff) {
						syncMode = syncModes::alert;
					}
					if (syncMode != syncModes::fine) {
						bool good = absFlowResult.Diff < avgImgDiff;
						for (auto d : flowBiasAbs) {
							if (d.distance(disp) > 20)
								good = false;
						}
						if (good) {
							// resync
							//tsf::print("Reset absolute (%v, %v)\n", disp.x, disp.y);
							didReset = true;
							syncMode = syncModes::fine;
							flowBias = disp;
						}
					}
					flowBiasAbs.erase(flowBiasAbs.begin());
					flowBiasAbs.push_back(disp);
					//flowBiasAbsRecover = disp;
					absDisp    = disp;
					absRestart = absRestartCheckInterval;
				} else {
					absRestart--;
				}
				Mesh mesh;
				SetupMesh(flat.Width, flat.Height, matchHeight, pixelsPerMeshCell, flow.MatchRadius, mesh);
				auto  flowResult = flow.Frame(mesh, Frustum(), flat, flatPrev, flowBias);
				auto  disp       = mesh.AvgValidDisplacement();
				float angle      = RAD2DEG * atan2(disp.y, disp.x);
				float angleNorm  = RAD2DEG * atan2(avgDir.y, avgDir.x);
				if (iFrame == 1) {
					avgDir     = disp.normalized();
					flowBias   = disp;
					avgImgDiff = flowResult.Diff;
				} else {
					float angleFromExpected = RAD2DEG * disp.angle(avgDir);
					if (fabs(angleFromExpected) > 5.0f) {
						// make sure we get a restart check soon, if our angle deviates substantially from the norm
						absRestart = min(absRestart, 15);
					}

					if (fabs(angle >= -90 - expectedAngleRange && angle <= -90 + expectedAngleRange)) {
						avgDir += 0.002f * (disp.normalized() - avgDir);
					}
					// Bad locks that I've seen go from a diff of 2426.7 to 17710.6. Wondering if that's not an overflow bug...... yeah.. probably overflow.
					if (flowResult.Diff < avgImgDiff * 2) {
						flowBias += 0.15f * (disp - flowBias);
						avgImgDiff += 0.05f * (flowResult.Diff - avgImgDiff);
						//if (fabs(flowBias.x) > 15.0f)
						//	flowBias.x *= 0.9f;
					}
				}
				tsf::print("%4.1f: %6.1f, %6.1f (%4.0f, %4.0f) (%6.1f, %6.1f) [%7.1f, %7.1f, %7.1f] %v\n", frameTime, disp.x, disp.y, angle, angleNorm, flowBias.x, flowBias.y, flowResult.Diff, avgImgDiff, absFlowResult.Diff, didReset ? "RESET" : "");
				//tsf::print("%6.1f, %6.1f (%6.1f) [%6.1f] %6.1f, %6.1f (%6.1f)\n", flowBias.x, flowBias.y, flowResult.Diff, absErr, absDisp.x, absDisp.y, absFlowResult.Diff);
				//tsf::print("%4.1f: %6.1f, %6.1f (%6.1f) [%6.1f] %6.1f, %6.1f (%6.1f) %s\n", frameTime, disp.x, disp.y, flowResult.Diff, absErr, absDisp.x, absDisp.y, absFlowResult.Diff, didReset ? "RESET" : "");

				speeds.push_back({frameTime, -flowBias.y});

				if (iFrame % 10 == 0 && outputFile != "stdout") {
					double relProcessingSpeed = frameTime / (time::Now() - startTime).Seconds();
					double remain             = (totalVideoSeconds - frameTime) / relProcessingSpeed;
					int    min                = (int) (remain / 60);
					int    sec                = (int) (remain - (min * 60));
					//tsf::print("\rTime remaining: %v:%02d", min, sec);
					//fflush(stdout);
				}
			}
			std::swap(flat, flatPrev);
			iFrame++;
		}
		// You might be tempted to add one frame worth of delay here to videoTimeOffset, but empirical measurements on our
		// Fuji X-T2 show that this formulation here is correct.
		videoTimeOffset += video.LastFrameTimeSeconds();
	} // for(ivideo)

	if (outputMode == SpeedOutputMode::JSON) {
		nlohmann::json j;
		j["time"] = firstVideoCreationTime.UnixNano() / 1000000;
		nlohmann::json jspeed;
		for (const auto& p : speeds) {
			nlohmann::json jp;
			jp.push_back(p.first);
			jp.push_back(p.second);
			jspeed.push_back(std::move(jp));
		}
		j["speeds"] = std::move(jspeed);
		auto js     = j.dump(4);
		fwrite(js.c_str(), js.size(), 1, outf);
	}

	fclose(outf);

	return Error();
}

int Speed2(argparse::Args& args) {
	auto videoFiles = strings::Split(args.Params[0], ',');
	auto err        = DoSpeed2(videoFiles, args.Has("csv") ? SpeedOutputMode::CSV : SpeedOutputMode::JSON, args.Get("outfile"));
	if (!err.OK()) {
		tsf::print(stderr, "Error: %v\n", err.Message());
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs