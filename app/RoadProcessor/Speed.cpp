#include "pch.h"

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

// Convert an RGBA image to an OpenCV RGB matrix (discard alpha)
static void RGBAToMat(int width, int height, int stride, const void* buf, cv::Mat& m) {
	if (m.rows != height || m.cols != width || m.type() != CV_8UC3)
		m.create(height, width, CV_8UC3);
	for (int y = 0; y < height; y++) {
		const char* src = (const char*) buf + y * stride;
		char*       dst = (char*) m.ptr(y);
		size_t      ww  = (size_t) width;
		for (size_t x = 0; x < ww; x++) {
			// RGB -> BGR (OpenCV is BGR)
			dst[0] = src[2];
			dst[1] = src[1];
			dst[2] = src[0];
			src += 4;
			dst += 3;
		}
	}
}

static cv::Mat RGBAToMat(int width, int height, int stride, const void* buf) {
	cv::Mat m;
	RGBAToMat(width, height, stride, buf, m);
	return m;
}

struct KeyPointSet {
	std::vector<cv::KeyPoint> Points;
	cv::Mat                   Descriptors;

	// Set points, detected via some function such as goodFeaturesToTrack
	void SetPoints(const std::vector<cv::Point2f>& points) {
		Points.resize(points.size());
		for (size_t i = 0; i < points.size(); i++)
			Points[i].pt = points[i];
	}
};

static void ComputeKeyPoints(cv::Mat img, KeyPointSet& kp) {
	bool orientNormalized = false;
	bool scaleNormalized  = true;
	auto featAlgo         = cv::xfeatures2d::FREAK::create(orientNormalized, scaleNormalized);

	vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(img, corners, 1000, 0.1, 5);
	kp.SetPoints(corners);
	//tsf::print("n: %v\n", corners.size());

	featAlgo->compute(img, kp.Points, kp.Descriptors);
}

static const double TooFast = 9999;

static double SpeedBetweenFramePair(cv::Mat img1, cv::Mat img2) {
	KeyPointSet kp1, kp2;
	ComputeKeyPoints(img1, kp1);
	ComputeKeyPoints(img2, kp2);

	// Compute keypoint matches, based only on feature matches (no position)
	vector<cv::DMatch>             initialMatches;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(kp1.Descriptors, kp2.Descriptors, initialMatches);

	// Compute final matches with GMS, which takes position into account
	vector<cv::DMatch> matchesGMS;
	cv::xfeatures2d::matchGMS(img1.size(), img2.size(), kp1.Points, kp2.Points, initialMatches, matchesGMS, false, false);

	//tsf::print("matchesGMS: %v\n", matchesGMS.size());

	double n        = 0;
	double sumDelta = 0;
	for (size_t i = 0; i < matchesGMS.size(); i++) {
		auto&  p1    = kp1.Points[matchesGMS[i].queryIdx];
		auto&  p2    = kp2.Points[matchesGMS[i].trainIdx];
		double delta = p2.pt.y - p1.pt.y;
		if (delta > 1000)
			tsf::print("(%v -> %v), %v,%v -> %v,%v\n", matchesGMS[i].trainIdx, matchesGMS[i].queryIdx, p1.pt.x, p1.pt.y, p2.pt.x, p2.pt.y);
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
					tsf::print("%.2f,%.3f\n", frameTime, speed);

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
		videoTimeOffset = video.LastFrameTimeSeconds();
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
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs