#include "pch.h"
#include "FeatureTracking.h"

// The "Speed" system analyzes inter-frame movement of keypoints, to determine our speed
// at which the vehicle is moving. We make no attempt to be physically accurate, because
// we do not need to be. All we need is a unit-less approximation of our speed.
// If the system is unable to match key points between a pair of frames, then it outputs
// a speed value of 0. If the system detects that the vehicle is standing still, then it
// outputs a speed value of 0.01. The max speed, at which the mechanism breaks down, is
// around 160, but this number depends on the resolution of the video, as well as the
// angle of the camera, and the framerate.

// To quickly test:
// build/run-roadprocessor -r speed --csv /home/ben/win/c/mldata/StellenboschFuji/4K-F2/DSCF1008.MOV 2>/dev/null

// (the 2>/dev/null is to silence the FFMpeg warnings that are typically found in camera video files)

// To generate JSON output in a file
// build/run-roadprocessor -r speed -o speed.json /home/ben/win/c/mldata/StellenboschFuji/4K-F2/DSCF1008.MOV

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

static const double TooFast = 9999;

static double SpeedBetweenFramePair(cv::Mat img1, cv::Mat img2) {
	int         maxKeyPoints = 1000;
	double      quality      = 0.1;
	double      minDistance  = 5;
	KeyPointSet kp1, kp2;
	ComputeKeyPoints(img1, maxKeyPoints, quality, minDistance, false, false, kp1);
	ComputeKeyPoints(img2, maxKeyPoints, quality, minDistance, false, false, kp2);

	vector<cv::DMatch> matches;
	ComputeMatch(img1.size(), img2.size(), kp1, kp2, false, false, matches);
	//tsf::print("matches: %v\n", matches.size());

	double n        = 0;
	double sumDelta = 0;
	for (size_t i = 0; i < matches.size(); i++) {
		auto&  p1    = kp1.Points[matches[i].queryIdx];
		auto&  p2    = kp2.Points[matches[i].trainIdx];
		double delta = p2.pt.y - p1.pt.y;
		if (delta > 1000)
			tsf::print("(%v -> %v), %v,%v -> %v,%v\n", matches[i].trainIdx, matches[i].queryIdx, p1.pt.x, p1.pt.y, p2.pt.x, p2.pt.y);
		sumDelta += delta;
		n++;
	}

	if (n == 0)
		return TooFast;

	return sumDelta / n;
}

static double Median(size_t n, double* x) {
	vector<double> xs;
	xs.resize(n);
	memcpy(&xs[0], x, n * sizeof(double));
	std::sort(xs.begin(), xs.end());
	return xs[xs.size() / 2];
}

enum class SpeedOutputMode {
	CSV,
	JSON,
};

static Error DoSpeed(vector<string> videoFiles, SpeedOutputMode outputMode, string outputFile) {
	FILE* outf = stdout;
	if (outputFile != "stdout") {
		outf = fopen(outputFile.c_str(), "w");
		if (!outf)
			return Error::Fmt("Failed to open output file '%v'", outputFile);
	}

	if (outputMode == SpeedOutputMode::CSV)
		tsf::print(outf, "time,speed\n");

	// Before we start this potentially lengthy process, make sure we can open every one of the video files specified.
	// Also, count their total time.
	double totalVideoSeconds = 0;
	for (const auto& v : videoFiles) {
		video::VideoFile video;
		auto             err = video.OpenFile(v);
		if (!err.OK())
			return err;
		totalVideoSeconds += video.GetVideoStreamInfo().DurationSeconds();
	}

	// speeds for every frame except for frame 0, as [time,speed]
	vector<pair<double, double>> speeds;

	auto   startTime       = time::Now();
	double videoTimeOffset = 0; // accumulating time counter, so that we can merge multiple videos into one timelime

	Error err;

	for (size_t ivideo = 0; ivideo < videoFiles.size(); ivideo++) {
		video::VideoFile video;
		err = video.OpenFile(videoFiles[ivideo]);
		if (!err.OK())
			return err;

		// We only monitor a cropped region of the image:
		// +----------+
		// |          |
		// |   +--+   |
		// |   |xx|   |
		// +---+--+---+
		// That region (indicated by xx) has the best speed-related movement inside it

		gfx::Rect32 crop;
		crop.x1 = video.Width() / 4;
		crop.x2 = video.Width() * 3 / 4;
		crop.y1 = video.Height() / 3;
		crop.y2 = video.Height();

		//video.SeekToMicrosecond(60 * 1000000);
		auto         info   = video.GetVideoStreamInfo();
		int          width  = info.Width;
		int          height = info.Height;
		int          stride = width * 4;
		void*        buf    = imqs_malloc_or_die(height * stride);
		cv::Mat      mTemp;
		cv::Mat      mPrev;
		cv::Mat      mNext;
		const size_t nrecent   = 5;
		double       recent[5] = {0};
		size_t       iFrame    = 0;

		while (err.OK()) {
			err = video.DecodeFrameRGBA(width, height, buf, stride);
			if (err == ErrEOF) {
				err = Error();
				break;
			} else if (!err.OK()) {
				break;
			}
			gfx::Image fullFrame(ImageFormat::RGBA, Image::ConstructWindow, stride, buf, width, height);
			gfx::Image cropImg = fullFrame.Window(crop.x1, crop.y1, crop.Width(), crop.Height());
			RGBAToMat(cropImg.Width, cropImg.Height, cropImg.Stride, cropImg.Data, mTemp);
			cv::cvtColor(mTemp, mNext, cv::COLOR_RGB2GRAY);
			if (iFrame >= 1) {
				// we need at least 2 frames
				double frameTime = videoTimeOffset + video.LastFrameTimeSeconds();
				double speed     = SpeedBetweenFramePair(mPrev, mNext);
				bool   tooFast   = speed == TooFast;
				if (!tooFast && speed == 0) {
					// set speed to some small epsilon value, when no movement detected, so that we don't end up confusing
					// "no data" with "stopped";
					speed = 0.01;
				}
				if (tooFast) {
					//speed = Median(nrecent, recent);
					speed = 0;
				} else {
					memmove(recent, recent + 1, sizeof(recent[0]) * (nrecent - 1));
					recent[nrecent - 1] = speed;
				}
				speeds.push_back({frameTime, speed});
				if (outputMode == SpeedOutputMode::CSV)
					tsf::print("%.3f,%.3f\n", frameTime, speed);

				if (iFrame % 10 == 0 && outputFile != "stdout") {
					double relProcessingSpeed = frameTime / (time::Now() - startTime).Seconds();
					double remain             = (totalVideoSeconds - frameTime) / relProcessingSpeed;
					int    min                = (int) (remain / 60);
					int    sec                = (int) (remain - (min * 60));
					tsf::print("\rTime remaining: %v:%02d", min, sec);
					fflush(stdout);
				}
			}
			std::swap(mPrev, mNext);
			iFrame++;
		}
		// You might be tempted to add one frame worth of delay here to videoTimeOffset, but empirical measurements on our
		// Fuji X-T2 show that this formulation here is correct.
		videoTimeOffset += video.LastFrameTimeSeconds();
		free(buf);
	} // for(ivideo)

	if (outputMode == SpeedOutputMode::JSON) {
		nlohmann::json j;
		for (const auto& p : speeds) {
			nlohmann::json jp;
			jp.push_back(p.first);
			jp.push_back(p.second);
			j.push_back(std::move(jp));
		}
		auto js = j.dump(4);
		fwrite(js.c_str(), js.size(), 1, outf);
	}

	fclose(outf);

	return err;
}

int Speed(argparse::Args& args) {
	auto videoFiles = strings::Split(args.Params[0], ',');
	auto err        = DoSpeed(videoFiles, args.Has("csv") ? SpeedOutputMode::CSV : SpeedOutputMode::JSON, args.Get("outfile"));
	if (!err.OK()) {
		// When trying to use CUDA, stdout seems to be dead at this point. Don't understand it.
		// But that's why we also output to stderr.
		tsf::print(stderr, "Error: %v\n", err.Message());
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs