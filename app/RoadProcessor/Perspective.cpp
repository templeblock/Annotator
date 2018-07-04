#include "pch.h"
#include "FeatureTracking.h"
#include "LensCorrection.h"
#include "Globals.h"
#include "Perspective.h"

/*
The functions in here compute the perspective factor Z2, from a video.
We assume that the camera is pointed down at a plane, which makes the
unprojection function extremely simple:
X = x/(Z1 - Z2*y)
Y = y/(Z1 - Z2*y)
where x and y are coordinates for the pixel that we're rendering, and
and X and Y are the source coordinates from the raw camera frame.
For an FHD image, the top-left corner is (-960, -540), and the bottom-right corner is (960, 540).

Z1 is a scaling factor, which we want at a value somewhere around 1.8 (although this
depends on Z2). If we keep Z1 at 1.0, then the resulting image maintains the resolution
at the top of the frame, but it shrinks the resolution at the bottom of the frame,
so we're losing information at the bottom of the frame. We don't want to lose resolution
at the bottom of the frame, so we rather make the entire resulting image larger. The pixels
at the top of the frame will then be stretched, but this is preferable to having the
pixels at the bottom be squashed.

The resulting unperspectified image looks like an upside-down triangle, or a V, but the bottom
of the V is cut off.. aka a frustum. This areas outside of the frustum are undefined. When we
choose our scaling factor, we measure it by the bottom of the frustum. If we want to lose zero
precision at the bottom of the frame, then we make sure that the bottom edge of the frustum
is exactly 1920 pixels across. We can apply any arbitrary factor on top of that, if we want
to sacrifice resolution, for performance, but that shouldn't be necessary.

*/

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

void Frustum::DebugPrintParams(float z1, float z2, int frameWidth, int frameHeight) const {
	tsf::print("%.4f %10f => %v x %v   %6.1f .. %6.1f (bottom scale %v, top scale %v)\n", z1, z2, Width, Height, X1, X2, (X2 - X1) / frameWidth, (float) Width / frameWidth);
}

// This runs newton's method, until it produces a y value that is
// within stopAtPrecision distance from desiredY, or until stopAtIteration iterations.
// Returns the best-performing x.
// dx is the epsilon for gradient computation.
float NewtonSearch(float x, float dx, float desiredY, float stopAtPrecision, int stopAtIteration, function<float(float x)> func) {
	for (int i = 0; i < stopAtIteration; i++) {
		float y = func(x);
		if (fabs(y - desiredY) <= stopAtPrecision)
			break;
		float y0   = func(x - dx);
		float y1   = func(x + dx);
		float grad = (y1 - y0) / (dx * 2);
		x          = x - (y - desiredY) / grad;
	}
	return x;
}

void FitQuadratic(const std::vector<std::pair<double, double>>& xy, double& a, double& b, double& c) {
	// http://mathforum.org/library/drmath/view/72047.html

	/*
	Now all you have to do is take your data (your eight points) and
	evaluate the various sums

			n
	Sj0 = sum x_i^j
			i=1

	for j = 0 through 4, and

			n
	Sj1 = sum x_i^j*y_i
			i=1

	for j = 0 through 2.  Then you substitute into the formulas for a, b,
	and c, and you are done!	
	*/

	double S[5][2];
	for (size_t j = 0; j < 5; j++) {
		S[j][0] = 0;
		S[j][1] = 0;
		for (size_t i = 0; i < xy.size(); i++) {
			S[j][0] += pow(xy[i].first, (double) j);
			S[j][1] += pow(xy[i].first, (double) j) * xy[i].second; // only used when j = 0,1,2
		}
	}

	/*
	Now we can use Cramer's Rule to give a, b, and c as formulas in these
	Sjk values.  They all have the same denominator:

		(S00*S20*S40 - S10^2*S40 - S00*S30^2 + 2*S10*S20*S30 - S20^3)	
	*/

	double denom = S[0][0] * S[2][0] * S[4][0] - pow(S[1][0], 2) * S[4][0] - S[0][0] * pow(S[3][0], 2) + 2 * S[1][0] * S[2][0] * S[3][0] - pow(S[2][0], 3);

	a = S[0][1] * S[1][0] * S[3][0] - S[1][1] * S[0][0] * S[3][0] - S[0][1] * pow(S[2][0], 2) + S[1][1] * S[1][0] * S[2][0] + S[2][1] * S[0][0] * S[2][0] - S[2][1] * pow(S[1][0], 2);
	b = S[1][1] * S[0][0] * S[4][0] - S[0][1] * S[1][0] * S[4][0] + S[0][1] * S[2][0] * S[3][0] - S[2][1] * S[0][0] * S[3][0] - S[1][1] * pow(S[2][0], 2) + S[2][1] * S[1][0] * S[2][0];
	c = S[0][1] * S[2][0] * S[4][0] - S[1][1] * S[1][0] * S[4][0] - S[0][1] * pow(S[3][0], 2) + S[1][1] * S[2][0] * S[3][0] + S[2][1] * S[1][0] * S[3][0] - S[2][1] * pow(S[2][0], 2);

	a /= denom;
	b /= denom;
	c /= denom;
}

// When we speak of normalized coordinates, we mean coordinates that are centered around the origin of the image

Vec2f FlatToCamera(float x, float y, float z1, float z2) {
	float z = z1 + z2 * y;
	return Vec2f(x / z, y / z);
}

// ux and uy are base-256
void FlatToCameraInt256(float z1, float z2, float x, float y, int32_t& u, int32_t& v) {
	float z  = z1 + z2 * y;
	float fx = x / z;
	float fy = y / z;
	fx *= 256;
	fy *= 256;
	u = (int32_t) fx;
	v = (int32_t) fy;
}

Vec2f FlatToCamera(int frameWidth, int frameHeight, float x, float y, float z1, float z2) {
	auto cam = FlatToCamera(x, y, z1, z2);
	cam.x += float(frameWidth / 2);
	cam.y += float(frameHeight / 2);
	return cam;
}

Vec2f CameraToFlat(Vec2f cam, float z1, float z2) {
	float yf = (cam.y * z1) / (1 - cam.y * z2);
	float xf = cam.x * (z1 + z2 * yf);
	return Vec2f(xf, yf);
}

Vec2f CameraToFlat(int frameWidth, int frameHeight, Vec2f cam, float z1, float z2) {
	cam.x    = cam.x - float(frameWidth / 2);
	cam.y    = cam.y - float(frameHeight / 2);
	float yf = (cam.y * z1) / (1 - cam.y * z2);
	float xf = cam.x * (z1 + z2 * yf);
	return Vec2f(xf, yf);
}

// Given a camera frame width and height, compute a frustum such that the bottom edge of the frustum
// has the exact same resolution as the bottom edge of the camera frame.
Frustum ComputeFrustum(int frameWidth, int frameHeight, float z1, float z2) {
	auto    topLeft  = CameraToFlat(frameWidth, frameHeight, Vec2f(0, 0), z1, z2);
	auto    topRight = CameraToFlat(frameWidth, frameHeight, Vec2f(frameWidth, 0), z1, z2);
	auto    botLeft  = CameraToFlat(frameWidth, frameHeight, Vec2f(0, frameHeight), z1, z2);
	auto    botRight = CameraToFlat(frameWidth, frameHeight, Vec2f(frameWidth, frameHeight), z1, z2);
	Frustum f;
	f.Width  = (int) (topRight.x - topLeft.x);
	f.Height = (int) (botLeft.y - topLeft.y);
	f.X1     = botLeft.x;
	f.X2     = botRight.x;
	return f;
}

// Use an iterative solution to find a value for Z1 (ie scale), which produces a 1:1 scale at
// the bottom of our flattened space.
float FindZ1ForIdentityScaleAtBottom(int frameWidth, int frameHeight, float z2) {
	auto scale = [=](float x) -> float {
		auto f = ComputeFrustum(frameWidth, frameHeight, x, z2);
		return (f.X2 - f.X1) / frameWidth;
	};
	return NewtonSearch(1.0, 0.001, 1.0, 1e-6, 50, scale);
}

static void PrintCamToFlat(int frameWidth, int frameHeight, float z1, float z2, Vec2f cam) {
	auto p = CameraToFlat(frameWidth, frameHeight, cam, z1, z2);
	tsf::print("(%5.0f,%5.0f) -> (%5.0f,%5.0f)\n", cam.x, cam.y, p.x, p.y);
}

static void Test(float z1, float z2) {
	int frameWidth  = 1920;
	int frameHeight = 1080;
	// compute raw numbers on z1, z2
	auto f = ComputeFrustum(frameWidth, frameHeight, z1, z2);
	f.DebugPrintParams(z1, z2, frameWidth, frameHeight);

	// auto compute frustum
	z1 = FindZ1ForIdentityScaleAtBottom(frameWidth, frameHeight, z2);
	f  = ComputeFrustum(frameWidth, frameHeight, z1, z2);
	f.DebugPrintParams(z1, z2, frameWidth, frameHeight);

	int w = frameWidth;
	int h = frameHeight;
	PrintCamToFlat(w, h, z1, z2, Vec2f(0, 0));
	PrintCamToFlat(w, h, z1, z2, Vec2f(w, 0));
	PrintCamToFlat(w, h, z1, z2, Vec2f(0, h));
	PrintCamToFlat(w, h, z1, z2, Vec2f(w, h));

	tsf::print("\n");
}

// Get ready to build up an image that has identical dimensions to 'camera', but with distortion correction
// applied. The output of this phase, is a 2D matrix, 1:1 with the original camera pixels, where every
// entry of the matrix is an XY pair, that points to the coordinates of the original, raw camera frame.
//float* fixedToRaw = (float*) imqs_malloc_or_die(camera.Width * camera.Height * 2);
// NOTE: We're using 16-bit coordinates here, and we allocate 12 bits for the pixel, and 4 bits for
// the sub-pixel. If this becomes too small (because camera res has increased), then rather use CUDA
// for this job, or switch to 32-bit coordinates. There is a bilinear lookup function for 32 bits,
// very similar to what's going on here for 16 bits.
// 4K res is 3840 x 2160. 12 bits of integer precision is enough for 4096, so we're OK here for 4K,
// but we'll see if 4 bits of sub-pixel precision is good enough.

// 12 + 4 = 16
const int DistortSubPixelBits = 4;
uint16_t* ComputeLensDistortionMatrix(int width, int height) {
	uint16_t* fixedToRaw = (uint16_t*) imqs_malloc_or_die(width * height * 2 * sizeof(uint16_t));

	for (int y = 0; y < height; y++) {
		global::Lens->ComputeDistortionForLine(y);
		const float* src = global::Lens->InterpPos;
		uint16_t*    dst = fixedToRaw + y * width * 2;
		float        fw  = (float) width;
		float        fh  = (float) height;
		for (int x = 0; x < width; x++) {
			// (01 Red) (23 Green) (45 Blue). Each pair is an XY coordinate, pointing into the raw image frame
			// We only use green, because we're not interested in chromatic abberations. The Lensfun docs say that
			// chromatic abberation fixing on lossy-compressed images (vs RAW) is just worse than nothing at all.
			float u = math::Clamp<float>(src[2], 0, fw);
			float v = math::Clamp<float>(src[3], 0, fw);
			dst[0]  = (uint16_t)(u * (1 << DistortSubPixelBits));
			dst[1]  = (uint16_t)(v * (1 << DistortSubPixelBits));
			dst += 2;
			src += 6;
		}
	}
	return fixedToRaw;
}

void RemovePerspective(const Image& camera, Image& flat, float z1, float z2, float originX, float originY) {
	uint16_t* fixedToRaw = ComputeLensDistortionMatrix(camera.Width, camera.Height);

	// Because our end product here is a frustum, we can compute straight lines down the edges, and
	// avoid sampling into those lines, which wins us a little bit of performance, because it's
	// less filtering that we need to do.
	auto  f     = ComputeFrustum(camera.Width, camera.Height, z1, z2);
	float x1Inc = (f.X1 + f.Width / 2) / (float) flat.Height;
	float x2Inc = (f.X2 + f.Width / 2 - flat.Width) / (float) flat.Height;
	// start the lines one pixel in. If we make this buffer large enough, then we don't need to
	// clamp inside our bilinear filter, which is a tiny speedup.
	float x1Edge = 1;
	float x2Edge = flat.Width - 1;

	// the -1 on the width is there because we're doing bilinear filtering, and we reach into sample+1
	// I'm not sure this is 100% correct.. and in fact we should probably add a bias of 1/2 a pixel
	int32_t srcClampU = (camera.Width - 1) * 256 - 1;
	int32_t srcClampV = (camera.Height - 1) * 256 - 1;

	// weird.. getting less distortion during optical flow calculations with less correction off
	bool doLensCorrection = false;

#pragma omp parallel for
	for (int y = 0; y < flat.Height; y++) {
		uint32_t* dst32 = (uint32_t*) flat.Data;
		dst32 += y * flat.Width;
		int     srcWidth = camera.Width;
		void*   src      = camera.Data;
		int32_t camHalfX = 256 * camera.Width / 2;
		int32_t camHalfY = 256 * camera.Height / 2;
		size_t  xStart   = (size_t) ceil(x1Edge + (float) y * x1Inc);
		size_t  xEnd     = (size_t) floor(x2Edge + (float) y * x2Inc);
		float   yM       = (float) y + originY;
		float   xM       = originX + xStart;
		for (size_t x = xStart; x < xEnd; x++, xM++) {
			int32_t u, v;
			// Flat to undistorted camera
			FlatToCameraInt256(z1, z2, xM, yM, u, v);
			u += camHalfX;
			v += camHalfY;
			if (doLensCorrection) {
				// undistorted camera to raw camera
				uint64_t fixed = raster::ImageBilinear_RG_U16(fixedToRaw, srcWidth, srcClampU, srcClampV, u, v);
				u              = fixed & 0xffff;
				v              = fixed >> 16;
				// bring the distortion parameters up to the required 8 bits of sub-pixel precision
				u = u << (8 - DistortSubPixelBits);
				v = v << (8 - DistortSubPixelBits);
			}
			// read from raw camera
			uint32_t color = raster::ImageBilinearRGBA(src, srcWidth, srcClampU, srcClampV, u, v);
			dst32[x]       = color;
		}
	}

	free(fixedToRaw);
}

struct ImageDiffResult {
	size_t MatchCount;
	double XMean;
	double YMean;
	double XVar;
	double YVar;
};

// Returns the difference between the two images, after they've been unprojected, and aligned
static ImageDiffResult ImageDiff(const Image& img1, const Image& img2, float z2) {
	bool debug = false;

	float z1         = FindZ1ForIdentityScaleAtBottom(img1.Width, img1.Height, z2);
	auto  f          = ComputeFrustum(img1.Width, img1.Height, z1, z2);
	auto  flatOrigin = CameraToFlat(img1.Width, img1.Height, Vec2f(0, 0), z1, z2);
	int   orgX       = (int) flatOrigin.x;
	int   orgY       = (int) flatOrigin.y;

	Image flat1, flat2;
	flat1.Alloc(ImageFormat::RGBA, f.Width, f.Height);
	flat2.Alloc(ImageFormat::RGBA, f.Width, f.Height);

	RemovePerspective(img1, flat1, z1, z2, orgX, orgY);
	RemovePerspective(img2, flat2, z1, z2, orgX, orgY);

	int   windowX1 = flat1.Width / 2 + f.X1 + 2;
	int   windowX2 = flat1.Width / 2 + f.X2 - 2;
	int   windowY1 = 0;
	int   windowY2 = flat1.Height;
	Image crop1, crop2;
	crop1          = flat1.Window(windowX1, windowY1, windowX2 - windowX1, windowY2 - windowY1);
	crop2          = flat2.Window(windowX1, windowY1, windowX2 - windowX1, windowY2 - windowY1);
	auto    mcrop1 = ImageToMat(crop1);
	auto    mcrop2 = ImageToMat(crop2);
	cv::Mat m1, m2;
	cv::cvtColor(mcrop1, m1, cv::COLOR_RGB2GRAY);
	cv::cvtColor(mcrop2, m2, cv::COLOR_RGB2GRAY);

	KeyPointSet        kp1, kp2;
	vector<cv::DMatch> matches;
	ComputeKeyPointsAndMatch("FREAK", m1, m2, 5000, 0.1, 10, true, false, kp1, kp2, matches);

	if (debug) {
		ImageIO imgIO;
		imgIO.SavePngFile("/home/ben/flat1.png", false, flat1.Width, flat1.Height, flat1.Stride, flat1.Data, 1);
		imgIO.SavePngFile("/home/ben/flat2.png", false, flat2.Width, flat2.Height, flat2.Stride, flat2.Data, 1);
		cv::Mat outImg;
		cv::drawMatches(mcrop1, kp1.Points, mcrop2, kp2.Points, matches, outImg);
		auto diag = MatToImage(outImg);
		imgIO.SavePngFile("/home/ben/match.png", false, diag.Width, diag.Height, diag.Stride, diag.Data, 1);
	}

	vector<double> xdelta, ydelta;
	for (const auto& m : matches) {
		auto& p1 = kp1.Points[m.queryIdx];
		auto& p2 = kp2.Points[m.trainIdx];
		xdelta.push_back(p2.pt.x - p1.pt.x);
		ydelta.push_back(p2.pt.y - p1.pt.y);
	}
	auto xstats = math::MeanAndVariance<double, double>(xdelta);
	auto ystats = math::MeanAndVariance<double, double>(ydelta);

	ImageDiffResult res;
	res.MatchCount = matches.size();
	res.XMean      = xstats.first;
	res.YMean      = ystats.first;
	res.XVar       = xstats.second;
	res.YVar       = ystats.second;

	return res;
}

static Error EstimateZ2(vector<string> videoFiles, float& bestZ2Estimate) {
	video::VideoFile video;
	auto             err = video.OpenFile(videoFiles[0]);
	if (!err.OK())
		return err;

	// we store this many pairs of frames in memory
	// A 1920x1080 RGBA image takes 8 MB memory. So 20 pairs is 316 MB, 100 pairs is 1.6 GB
	int nSamples = 100;

	// If we have less than this number of valid (ie decent quality) frame pairs, then the entire process is considered a failure
	int minValidSamples = 60;

	// seconds apart each sample
	double spacing = video.GetVideoStreamInfo().DurationSeconds() / (nSamples + 1);

	int width  = video.Width();
	int height = video.Height();

	vector<pair<Image, Image>> samples;
	for (int i = 0; i < nSamples; i++) {
		//auto err = video.SeekToSecond(40);
		auto err = video.SeekToSecond(i * spacing);
		if (!err.OK())
			return err;
		pair<Image, Image> sample;
		sample.first.Alloc(gfx::ImageFormat::RGBA, width, height);
		sample.second.Alloc(gfx::ImageFormat::RGBA, width, height);
		err = video.DecodeFrameRGBA(width, height, sample.first.Data, sample.second.Stride);
		err |= video.DecodeFrameRGBA(width, height, sample.second.Data, sample.second.Stride);
		if (!err.OK())
			return err;
		samples.push_back(std::move(sample));
	}

	// initial search parameters:
	// z2 = -0.0011,  7 iterations, step with z2 += 0.0001
	// z2 = -0.0011, 14 iterations, step with z2 += 0.00005

	float z2_min = -0.0011;
	float z2_max = -0.0005;
	//float z2 = -0.001;
	//float z2 = -0.001050;
	//float z2 = -0.000750;

	// pairs of z2 and X match variance, which we fit a parabola to.
	vector<pair<double, double>> z2AndVar;
	/*
	z2AndVar = {
	    {-0.001100, 268.786},
	    {-0.001050, 211.148},
	    {-0.001000, 183.983},
	    {-0.000950, 147.988},
	    {-0.000900, 151.185},
	    {-0.000850, 149.401},
	    {-0.000800, 168.865},
	    {-0.000750, 190.066},
	    {-0.000700, 231.291},
	    {-0.000650, 267.394},
	    {-0.000600, 334.534},
	    {-0.000550, 390.127},
	    {-0.000500, 473.246},
	    {-0.000450, 534.972},
	};
	*/

	// count the number of times we've noticed that in this frame the vehicle is not moving fast enough,
	// and also how many times it had too few matches
	vector<int> tooSlow;
	vector<int> tooFew;
	for (size_t i = 0; i < samples.size(); i++) {
		tooFew.push_back(0);
		tooSlow.push_back(0);
	}

	int niter = 20;
	for (int iter = 0; iter < niter; iter++) {
		float z2 = z2_min + iter * (z2_max - z2_min) / (niter - 1);
		float z1 = FindZ1ForIdentityScaleAtBottom(samples[0].first.Width, samples[0].first.Height, z2);
		auto  f  = ComputeFrustum(samples[0].first.Width, samples[0].first.Height, z1, z2);
		//double score   = 0;
		//double nsample = 0;
		vector<Vec2d> stats;
		for (size_t i = 0; i < samples.size(); i++) {
			// once we've seen a frame as been "too slow" or "too few" too often, skip it forever
			if (tooFew[i] >= 3 || tooSlow[i] >= 3) {
				continue;
			}
			const auto& sample = samples[i];
			auto        diff   = ImageDiff(sample.first, sample.second, z2);
			if (diff.MatchCount < 100) {
				tooFew[i]++;
			} else {
				//score += diff.XVar;
				//nsample++;
				//tsf::print("%3d, %5d, X:(%8.3f, %8.3f), Y(%8.3f, %8.3f)\n", i, diff.MatchCount, diff.XMean, diff.XVar, diff.YMean, diff.YVar);
				// We need to be moving forward, and YMean predicts this. Typical road speed is around 100 to 300 pixels.
				if (diff.YMean < 20) {
					tooSlow[i]++;
				} else {
					stats.emplace_back(diff.XVar, diff.YVar);
				}
			}
		}
		if (stats.size() < minValidSamples)
			return Error::Fmt("Too few valid frames (z1 = %.8f, frames = %v/%v)", z1, stats.size(), samples.size());
		sort(stats.begin(), stats.end(), [](Vec2d a, Vec2d b) { return a.size() < b.size(); });
		double score = 0;
		size_t burn  = stats.size() / 10;
		for (size_t i = burn; i < stats.size() - burn; i++) {
			score += stats[i].x;
		}
		double nsample = stats.size() - 2 * burn;
		score /= nsample;
		z2AndVar.emplace_back(z2, score);
		tsf::print("%.6f,%.3f,%v,%v,%v\n", z2, score, nsample, f.Width, f.Height);
		//z2 += 0.00005;
	}

	// scale and bias the z2 values so that the quadratic fit has better conditioned numbers
	float z2FitBias  = 0.0009;
	float z2FitScale = 1000000;
	for (auto& p : z2AndVar)
		p.first = (p.first + z2FitBias) * z2FitScale;

	double a, b, c;
	FitQuadratic(z2AndVar, a, b, c);
	double bestZ2  = -b / (2 * a);
	bestZ2         = bestZ2 / z2FitScale - z2FitBias;
	bestZ2Estimate = (float) bestZ2;
	//tsf::print("%v %v %v -> %v\n", a, b, c, bestZ2);

	return Error();
}

static Error DoPerspective(vector<string> videoFiles) {
	float z2  = 0;
	auto  err = EstimateZ2(videoFiles, z2);
	if (!err.OK())
		return err;

	// 3023.mov -> 0.00283583 -0.115201 160.89 -> -0.000879688
	// z2 = -0.000879688;
	return Error();
}

int Perspective(argparse::Args& args) {
	//Test(1, -0.0007);
	auto videoFiles = strings::Split(args.Params[0], ',');
	auto err        = DoPerspective(videoFiles);
	if (!err.OK()) {
		tsf::print("Error: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs