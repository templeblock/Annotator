#include "pch.h"
#include "FeatureTracking.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

void KeyPointSet::SetPoints(const std::vector<cv::Point2f>& points) {
	Points.resize(points.size());
	for (size_t i = 0; i < points.size(); i++)
		Points[i].pt = points[i];
}

void RGBAToMat(int width, int height, int stride, const void* buf, cv::Mat& m) {
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

cv::Mat RGBAToMat(int width, int height, int stride, const void* buf) {
	cv::Mat m;
	RGBAToMat(width, height, stride, buf, m);
	return m;
}

cv::Mat ImageToMat(const gfx::Image& img) {
	return RGBAToMat(img.Width, img.Height, img.Stride, img.Data);
}

gfx::Image MatToImage(cv::Mat mat) {
	Image img;
	img.Alloc(ImageFormat::RGBA, mat.cols, mat.rows);
	for (int y = 0; y < img.Height; y++) {
		const uint8_t* src = (const uint8_t*) mat.ptr(y);
		uint8_t*       dst = (uint8_t*) img.Data + y * img.Stride;
		int            w   = img.Width;
		if (mat.type() == CV_8UC1) {
			for (int x = 0; x < w; x++) {
				dst[0] = src[0];
				dst[1] = src[0];
				dst[2] = src[0];
				dst[3] = 255;
				src += 1;
				dst += 4;
			}
		} else if (mat.type() == CV_8UC3) {
			for (int x = 0; x < w; x++) {
				// BGR -> RGB (OpenCV is BGR)
				dst[0] = src[2];
				dst[1] = src[1];
				dst[2] = src[0];
				dst[3] = 255;
				src += 3;
				dst += 4;
			}
		} else {
			IMQS_DIE();
		}
	}
	return img;
}

void ComputeKeyPoints(cv::Mat img, int maxPoints, double quality, double minDistance, bool orientNormalized, bool scaleNormalized, KeyPointSet& kp) {
	auto featAlgo = cv::xfeatures2d::FREAK::create(orientNormalized, scaleNormalized);

	vector<cv::Point2f> corners;
	//cv::goodFeaturesToTrack(img, corners, maxPoints, quality, minDistance, cv::noArray(), 3, true);
	cv::goodFeaturesToTrack(img, corners, maxPoints, quality, minDistance, cv::noArray(), 8);
	kp.SetPoints(corners);
	//tsf::print("n: %v\n", corners.size());

	featAlgo->compute(img, kp.Points, kp.Descriptors);
}

void ComputeMatch(cv::Size img1Size, cv::Size img2Size, const KeyPointSet& kp1, const KeyPointSet& kp2, bool withRotation, bool withScale, std::vector<cv::DMatch>& matches) {
	// Compute keypoint matches, based only on feature matches (no position)
	vector<cv::DMatch>             initialMatches;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(kp1.Descriptors, kp2.Descriptors, initialMatches);

	// Compute final matches with GMS, which takes position into account
	cv::xfeatures2d::matchGMS(img1Size, img2Size, kp1.Points, kp2.Points, initialMatches, matches, withRotation, withScale);
}

void ComputeKeyPointsAndMatch(cv::Mat img1, cv::Mat img2, int maxPoints, double quality, double minDistance, bool withRotation, bool withScale, KeyPointSet& kp1, KeyPointSet& kp2, std::vector<cv::DMatch>& matches) {
	ComputeKeyPoints(img1, maxPoints, quality, minDistance, withRotation, withScale, kp1);
	ComputeKeyPoints(img2, maxPoints, quality, minDistance, withRotation, withScale, kp2);
	ComputeMatch(img1.size(), img2.size(), kp1, kp2, withRotation, withScale, matches);
}

} // namespace roadproc
} // namespace imqs