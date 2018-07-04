#include "pch.h"
#include "FeatureTracking.h"
#include "Globals.h"
#include "Perspective.h"

// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 2 --start 0 /home/ben/win/c/mldata/DSCF3023.MOV -0.000879688
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' stitch -n 10 --start 5 /home/ben/win/c/mldata/DSCF3023.MOV -0.000879688

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

// Compute the Mean Squared Error between img1 and img2, for the rectangle defined
// by rect1, in the coordinate space of img1. The rectangle is translated by dx
// and dy, before being applied to img2.
// The translation caused by dx and dy must be valid - ie it may not cause the
// rectangle to be outside the bounds of img2.
// The result is the square root of the average of the sum of the squares of the differences of
// each pixel.
//static float DiffMSE(const Image& img1, const Image& img2, Rect32 rect1, int dx, int dy) {
static double DiffAvg(const Image& img1, const Image& img2, Rect32 rect1, int dx, int dy) {
	IMQS_ASSERT(img1.Format == ImageFormat::Gray);
	IMQS_ASSERT(img2.Format == ImageFormat::Gray);
	IMQS_ASSERT(rect1.x1 + dx >= 0);
	IMQS_ASSERT(rect1.x2 + dx <= img2.Width);
	IMQS_ASSERT(rect1.y1 + dy >= 0);
	IMQS_ASSERT(rect1.y2 + dy <= img2.Height);
	int     w   = rect1.Width();
	int     h   = rect1.Height();
	int64_t sum = 0;
	for (int y = 0; y < h; y++) {
		const uint8_t* p1 = img1.At(rect1.x1, rect1.y1 + y);
		const uint8_t* p2 = img2.At(rect1.x1 + dx, rect1.y1 + dy + y);
		for (int x = 0; x < w; x++) {
			//int d = (int) p1[x] - (int) p2[x];
			//sum += d * d;
			int d = (int) p1[x] - (int) p2[x];
			sum += d < 0 ? -d : d;
		}
	}
	return (double) sum / (double) (w * h);
	//return (float) sqrt((double) sum / (double) (w * h));
}

static double ImageStdDev(const Image& img, Rect32 crop) {
	uint32_t sum = 0;
	for (int y = crop.y1; y < crop.y2; y++) {
		const uint8_t* p = img.At(crop.x1, y);
		for (int x = crop.x1; x < crop.x2; x++, p++)
			sum += *p;
	}
	sum /= crop.Width() * crop.Height();
	uint32_t var = 0;
	for (int y = crop.y1; y < crop.y2; y++) {
		const uint8_t* p = img.At(crop.x1, y);
		for (int x = crop.x1; x < crop.x2; x++, p++) {
			var += ((uint32_t) *p - sum) * ((uint32_t) *p - sum);
		}
	}
	double dvar = (double) var;
	dvar        = sqrt(dvar / (double) (crop.Width() * crop.Height()));
	return dvar;
}

class OpticalFlow {
public:
	OpticalFlow() {
		LastGridDx.resize(GridW * GridH);
		LastGridDy.resize(GridW * GridH);
	}

	// We compute the transform from img1 to img2
	void Frame(Image& img1, Image& img2) {
		int   frameNumber = HistorySize++;
		Image img1G       = img1.AsType(ImageFormat::Gray);
		Image img2G       = img2.AsType(ImageFormat::Gray);
		//img1G.SavePng("img1G.png");
		//img2G.SavePng("img2G.png");

		// tiny images are 1/4 of the original image size
		//Image tiny1         = img1G.HalfSizeCheap();
		//Image tiny2         = img2G.HalfSizeCheap();
		Image tiny1    = img1G;
		Image tiny2    = img2G;
		int   downIter = 0;
		//int   tinyDownscale = 1 << (1 + downIter);
		int tinyDownscale = 1;
		for (int iter = 0; iter < downIter; iter++)
			tiny1 = tiny1.HalfSizeCheap();
		for (int iter = 0; iter < downIter; iter++)
			tiny2 = tiny2.HalfSizeCheap();

		//tiny1.BoxBlur(1, 3);
		//tiny2.BoxBlur(1, 3);
		tiny1.SavePng("tiny1.png");
		tiny2.SavePng("tiny2.png");

		int absMinVSearch = -40 / tinyDownscale; // negative: driving backwards
		int absMaxVSearch = 300 / tinyDownscale; // positive: driving forwards
		int absMinHSearch = -32 / tinyDownscale;
		int absMaxHSearch = 32 / tinyDownscale;

		int minVSearch = absMinVSearch;
		int maxVSearch = absMaxVSearch;
		int minHSearch = absMinHSearch;
		int maxHSearch = absMaxHSearch;

		//if (frameNumber >= 3) {
		//	minVSearch = LastDy - 20 / tinyDownscale;
		//	maxVSearch = LastDy + 20 / tinyDownscale;
		//	hSearch    = LastDx + 10 / tinyDownscale;
		//	minVSearch = max(minVSearch, absMinVSearch);
		//	maxVSearch = min(maxVSearch, absMaxVSearch);
		//}

		int            smallW = 16;
		vector<double> allDx, allDy;
		vector<double> gridDx, gridDy;
		gridDx.resize(GridW * GridH);
		gridDy.resize(GridW * GridH);
		double avgSD   = 0;
		double avgDiff = 0;
		for (int gx = 0; gx < GridW; gx++) {
			for (int gy = 0; gy < GridH; gy++) {
				if (frameNumber >= 3) {
					minVSearch = (int) (GridEl(LastGridDy, gx, gy) - 20.0 / (double) tinyDownscale);
					maxVSearch = (int) (GridEl(LastGridDy, gx, gy) + 20.0 / (double) tinyDownscale);
					minHSearch = (int) (GridEl(LastGridDx, gx, gy) - 10.0 / (double) tinyDownscale);
					maxHSearch = (int) (GridEl(LastGridDx, gx, gy) + 10.0 / (double) tinyDownscale);
					minHSearch = max(minHSearch, absMinHSearch);
					maxHSearch = min(maxHSearch, absMaxHSearch);
					minVSearch = max(minVSearch, absMinVSearch);
					maxVSearch = min(maxVSearch, absMaxVSearch);
				}
				int            x = -absMinHSearch + gx * (tiny1.Width + absMinHSearch - absMaxHSearch - smallW) / (GridW - 1);
				int            y = -absMinVSearch + gy * (tiny1.Height + absMinVSearch - absMaxVSearch - smallW) / (GridH - 1);
				Rect32         rect(x, y, x + smallW, y + smallW);
				vector<double> allD;
				double         bestD  = 1e10;
				int            bestDx = 0;
				int            bestDy = 0;
				for (int dy = minVSearch; dy <= maxVSearch; dy++) {
					for (int dx = minHSearch; dx <= maxHSearch; dx++) {
						double d = DiffAvg(tiny1, tiny2, rect, dx, dy);
						allD.push_back(d);
						if (d < bestD) {
							bestDx = dx;
							bestDy = dy;
							bestD  = d;
						}
					}
				}
				//auto mv = math::MeanAndVariance<double, double>(allD);
				auto mm = math::MinMax<double, double>(allD);
				auto sd = ImageStdDev(tiny1, rect);
				avgSD += sd / (double) (GridW * GridH);
				avgDiff += bestD / (double) (GridW * GridH);
				//tsf::print("%3d %3d: %5.2f %5.2f, %5.2f .. %5.2f, %4d %4d\n", gx, gy, mv.first, mv.second, mm.first, mm.second, bestDx, bestDy);
				//tsf::print("%3d %3d: %5.2f, %5.2f .. %5.2f, %4d %4d\n", gx, gy, sd, mm.first, mm.second, bestDx, bestDy);
				allDx.push_back(bestDx);
				allDy.push_back(bestDy);
				GridEl(gridDx, gx, gy) = bestDx;
				GridEl(gridDy, gx, gy) = bestDy;
			}
		}

		pdqsort(allDx.begin(), allDx.end());
		pdqsort(allDy.begin(), allDy.end());
		//double finalDx = allDx[allDx.size() / 2];
		//double finalDy = allDy[allDy.size() / 2];
		auto   allDxMV = math::MeanAndVariance<double, double>(5, &allDx[allDx.size() - 5]);
		auto   allDyMV = math::MeanAndVariance<double, double>(5, &allDy[allDy.size() - 5]);
		double finalDx = allDxMV.first;
		double finalDy = allDyMV.first;
		tsf::print("%.2f,%.2f,%.2f,%.2f,%.2f\n", (double) frameNumber / 29.97, finalDy, finalDx, avgSD, avgDiff);

		// might need to compute max vehicle accel/break speed to know what this number 'a' should be
		double a = frameNumber == 0 ? 1 : 0.1;
		LastDx   = LastDx + a * (finalDx - LastDx);
		LastDy   = LastDy + a * (finalDy - LastDy);
		// Update running averages of grid offsets
		for (int gy = 0; gy < GridH; gy++) {
			for (int gx = 0; gx < GridW; gx++) {
				GridEl(LastGridDx, gx, gy) += a * (GridEl(gridDx, gx, gy) - GridEl(LastGridDx, gx, gy));
				GridEl(LastGridDy, gx, gy) += a * (GridEl(gridDy, gx, gy) - GridEl(LastGridDy, gx, gy));
			}
		}
		//PrintGrid(LastGridDy);

		/*
	int minVerticalInitialSearch = -40 / tinyDownscale; // negative: driving backwards
	int maxVerticalInitialSearch = 300 / tinyDownscale; // positive: driving forwards
	int initialHorizontalSearch  = 32 / tinyDownscale;

	// half-size of the window in the center of the image, that we initially match.
	int initialSearchWidth  = min(400, tiny1.Width / 2 - initialHorizontalSearch);
	int initialSearchHeight = 30;

	vector<float> centerDy;
	Rect32        rect(tiny1.Width / 2, tiny1.Height / 2, tiny1.Width / 2, tiny1.Height / 2);
	rect.Expand(initialSearchWidth, initialSearchHeight);
	for (int dy = minVerticalInitialSearch; dy <= maxVerticalInitialSearch; dy++) {
		float minD = 1e10;
		for (int dx = initialHorizontalSearch; dx <= initialHorizontalSearch; dx++) {
			float d = DiffMSE(tiny1, tiny2, rect, dx, dy);
			minD    = min(minD, d);
		}
		centerDy.push_back(minD);
	}
	float minVal = 1e10;
	float speed;
	for (size_t i = 0; i < centerDy.size(); i++) {
		tsf::print("%d,%5.5f\n", minVerticalInitialSearch + (int) i, centerDy[i]);
		//tsf::print("%4d: %5.5f\n", minVerticalInitialSearch + (int) i, centerDy[i]);
		if (centerDy[i] < minVal) {
			minVal = centerDy[i];
			speed  = i;
		}
	}
	double sec     = frameNumber / 29.97;
	int    minute  = (int) (sec / 60);
	int    second  = (int) (sec - (minute * 60));
	int    msecond = (int) ((sec - (minute * 60) - second) * 1000);
	//tsf::print("%2d:%02d.%03d,%.5f\n", minute, second, msecond, speed);
	tsf::print("%05d,%.5f\n", frameNumber, speed);
	*/
	}

private:
	int                 GridW       = 5;
	int                 GridH       = 5;
	double              LastDx      = 0;
	double              LastDy      = 0;
	int                 HistorySize = 0;
	std::vector<double> LastGridDx;
	std::vector<double> LastGridDy;

	double& GridEl(std::vector<double>& grid, int x, int y) { return grid[y * GridW + x]; }

	void PrintGrid(std::vector<double>& grid) {
		for (int y = 0; y < GridH; y++) {
			for (int x = 0; x < GridW; x++) {
				tsf::print("%3.0f ", GridEl(grid, x, y));
			}
			tsf::print("\n");
		}
	}
};

static void StitchFrames(OpticalFlow& flow, int frameNumber, Frustum f, Image& img1, Image& img2) {
	int windowWidth  = f.X2 - f.X1 - 2;
	int windowTop    = 300;
	int windowHeight = img1.Height - windowTop;
	int windowLeft   = (img1.Width - windowWidth) / 2;

	auto img1Crop = img1.Window(windowLeft, windowTop, windowWidth, windowHeight);
	auto img2Crop = img2.Window(windowLeft, windowTop, windowWidth, windowHeight);

	flow.Frame(img1Crop, img2Crop);
	//ComputeOpticalFlow(frameNumber, img1Crop, img2Crop);

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

	OpticalFlow flow;

	for (int i = 0; i < count; i++) {
		err = video.DecodeFrameRGBA(img.Width, img.Height, img.Data, img.Stride);
		if (err == ErrEOF)
			break;
		if (!err.OK())
			return err;

		RemovePerspective(img, flat, z1, z2, (int) flatOrigin.x, (int) flatOrigin.y);
		if (i != 0)
			StitchFrames(flow, i, f, flatPrev, flat);

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