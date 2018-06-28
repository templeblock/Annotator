#pragma once

namespace imqs {
namespace roadproc {

struct KeyPointSet {
	std::vector<cv::KeyPoint> Points;
	cv::Mat                   Descriptors;

	// Set points, detected via some function such as goodFeaturesToTrack
	void SetPoints(const std::vector<cv::Point2f>& points);
};

void       RGBAToMat(int width, int height, int stride, const void* buf, cv::Mat& m); // Returns an RGB matrix (discards alpha)
cv::Mat    RGBAToMat(int width, int height, int stride, const void* buf);             // Returns an RGB matrix (discards alpha)
cv::Mat    ImageToMat(const gfx::Image& img);                                         // Returns an RGB matrix (discards alpha)
gfx::Image MatToImage(cv::Mat mat);
void       ComputeKeyPoints(cv::Mat img, int maxPoints, double quality, double minDistance, bool orientNormalized, bool scaleNormalized, KeyPointSet& kp);
void       ComputeMatch(cv::Size img1Size, cv::Size img2Size, const KeyPointSet& kp1, const KeyPointSet& kp2, bool withRotation, bool withScale, std::vector<cv::DMatch>& matches);
void       ComputeKeyPointsAndMatch(cv::Mat img1, cv::Mat img2, int maxPoints, double quality, double minDistance, bool withRotation, bool withScale, KeyPointSet& kp1, KeyPointSet& kp2, std::vector<cv::DMatch>& matches);

} // namespace roadproc
} // namespace imqs