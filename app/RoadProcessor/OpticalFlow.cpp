#include "pch.h"
#include "OpticalFlow.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

// NOTE: If we end up struggling with the shadow of the car, then we should try using linear light,
// during the LocalContrast step.

// this is just to make things easier to see when debugging images, but results are identical. we actually want 1x scale to minimize clipping
const int DebugBrightenLocalContrast = 1;

static void    LocalContrast(Image& img, int size, int iterations);
static int32_t DiffSum(const Image& img1, const Image& img2, Rect32 rect1, Rect32 rect2);
static double  ImageStdDev(const Image& img, Rect32 crop);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

OpticalFlow::OpticalFlow() {
}

static Rect32 MakeBoxAroundPoint(int x, int y, int radius) {
	return Rect32(x - radius, y - radius, x + radius, y + radius);
}

DeltaGrid::DeltaGrid() {
}

DeltaGrid::DeltaGrid(DeltaGrid& g) {
	*this = g;
}

DeltaGrid& DeltaGrid::operator=(const DeltaGrid& g) {
	if (this != &g) {
		Alloc(g.Width, g.Height);
		memcpy(Delta, g.Delta, sizeof(Vec2f) * Width * Height);
		memcpy(Valid, g.Valid, sizeof(bool) * Width * Height);
	}
	return *this;
}

DeltaGrid::~DeltaGrid() {
	free(Delta);
	free(Valid);
}

void DeltaGrid::Alloc(int w, int h) {
	if (w != Width || h != Height) {
		free(Delta);
		free(Valid);
		Width  = w;
		Height = h;
		Delta  = (gfx::Vec2f*) imqs_malloc_or_die(w * h * sizeof(gfx::Vec2f));
		Valid  = (bool*) imqs_malloc_or_die(w * h * sizeof(bool));
	}
}

static void CopyMeshToDelta(const Mesh& m, Rect32 rect, DeltaGrid& g, Vec2f norm) {
	g.Alloc(rect.Width(), rect.Height());
	//Vec2f norm = m.At(rect.x1, rect.y1).Pos - m.At(rect.x1, rect.y1).UV;
	for (int y = rect.y1; y < rect.y2; y++) {
		for (int x = rect.x1; x < rect.x2; x++) {
			g.At(x - rect.x1, y - rect.y1)      = m.At(x, y).Pos - m.At(x, y).UV - norm;
			g.IsValid(x - rect.x1, y - rect.y1) = m.At(x, y).IsValid;
		}
	}
}

static void CopyDeltaToMesh(const DeltaGrid& g, Mesh& m, Rect32 rect, Vec2f norm) {
	//Vec2f norm = m.At(rect.x1, rect.y1).Pos - m.At(rect.x1, rect.y1).UV;

	for (int y = rect.y1; y < rect.y2; y++) {
		for (int x = rect.x1; x < rect.x2; x++) {
			m.At(x, y).Pos     = g.At(x - rect.x1, y - rect.y1) + m.At(x, y).UV + norm;
			m.At(x, y).IsValid = g.IsValid(x - rect.x1, y - rect.y1);
		}
	}
}

static bool CompareVec2X(const Vec2f& a, const Vec2f& b) {
	return a.x < b.x;
}

static bool CompareVec2Y(const Vec2f& a, const Vec2f& b) {
	return a.y < b.y;
}

struct ClusterStat {
	Vec2f Center    = Vec2f(0, 0);
	int   Count     = 0;
	float Tightness = 0;
	// typical decent values for tightness are around 5, so an epsilon of 0.1 feels about right
	float Alpha() const { return (float) Count / (Tightness + 0.1); }
};

static void ComputeClusterStats(const vector<cv::Point2f>& allCV, const vector<int>& cluster, const vector<cv::Point2f>& clusterCenters, vector<ClusterStat>& stats) {
	stats.resize(clusterCenters.size());
	for (size_t i = 0; i < clusterCenters.size(); i++) {
		stats[i]          = ClusterStat();
		stats[i].Center.x = clusterCenters[i].x;
		stats[i].Center.y = clusterCenters[i].y;
	}

	for (size_t i = 0; i < cluster.size(); i++) {
		int   c  = cluster[i];
		Vec2f cv = Vec2f(clusterCenters[c].x, clusterCenters[c].y);
		stats[c].Count++;
		stats[c].Tightness += cv.distance2D(Vec2f(allCV[i].x, allCV[i].y));
	}
	for (size_t i = 0; i < clusterCenters.size(); i++) {
		stats[i].Tightness /= (float) stats[i].Count;
	}
}

static void DumpKMeans(vector<ClusterStat> clusters) {
	tsf::print("%v clusters:\n", clusters.size());
	for (auto c : clusters) {
		tsf::print("  Alpha: %4.2f Count: %2d, Tightness: %.1f, Center: %5.1f, %5.1f\n", c.Alpha(), c.Count, c.Tightness, c.Center.x, c.Center.y);
	}
}

// Returns the number of points that were replaced with a filtered replica
// I tried a smaller filter size of 3x3, but it easily introduces noisy samples
// into the final dataset. This is partly due to the sloppy metric "maxGlobalDistance".
static int MedianFilter(int pass, DeltaGrid& g, bool& hasMassiveOutliers) {
	DeltaGrid gnew                  = g;
	hasMassiveOutliers              = false;
	const int filterRadius          = 2;
	const int filterSize            = 2 * filterRadius + 1;
	const int filterSizeSQ          = filterSize * filterSize;
	float     maxDistance           = 2;  // If sample is more than maxDistance from local filter estimate, then it is filtered
	float     maxGlobalDistanceSoft = 15; // If sample is more than maxGlobalDistanceSoft from global distance estimate, then it is filtered
	float     maxGlobalDistanceHard = 25; // If sample is more than maxGlobalDistanceHard from global distance estimate, then it is replaced with the global estimate
	int       nrep                  = 0;

	// compute global median, so that we can throw away extreme outliers
	Vec2f               globalEstimate;
	vector<Vec2f>       all;
	vector<cv::Point2f> allCV;
	all.reserve(g.Width * g.Height);
	for (int y = 0; y < g.Height; y++) {
		for (int x = 0; x < g.Width; x++) {
			Vec2f p = g.At(x, y);
			all.push_back(p);
			allCV.push_back(cv::Point2f(p.x, p.y));
		}
	}
	pdqsort(all.begin(), all.end(), CompareVec2X);
	globalEstimate.x = all[all.size() / 2].x;
	pdqsort(all.begin(), all.end(), CompareVec2Y);
	globalEstimate.y = all[all.size() / 2].y;

	// after the first pass, we can skip this expensive step
	if (pass == 0) {
		bool                debugKMeans = false;
		vector<int>         icluster;
		vector<cv::Point2f> clusterCenters;
		vector<float>       clusterTightness;
		vector<ClusterStat> clusters;
		cv::TermCriteria    term(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1.0);
		if (debugKMeans) {
			for (int ncluster = 2; ncluster <= 6; ncluster++) {
				cv::kmeans(allCV, ncluster, icluster, term, 1, 0, clusterCenters);
				ComputeClusterStats(allCV, icluster, clusterCenters, clusters);
				DumpKMeans(clusters);
			}
			exit(0);
		} else {
			// From one video that I looked at, 5 seems to be a sweet spot
			int nclusters = 5;
			cv::kmeans(allCV, nclusters, icluster, term, 3, 0, clusterCenters);
			ComputeClusterStats(allCV, icluster, clusterCenters, clusters);
			float       maxAlpha = 0;
			ClusterStat best;
			for (const auto& c : clusters) {
				// This constant of 5% is a thumbsuck, given the typical number of points that we align
				int minCount = int(0.05 * (double) allCV.size());
				if (c.Count > minCount && c.Alpha() > maxAlpha) {
					maxAlpha = c.Alpha();
					best     = c;
				}
			}
			globalEstimate = best.Center;
		}
	}
	//tsf::print("%3d. %5.1f,%5.1f\n", pass, globalEstimate.x, globalEstimate.y);

	// replace obvious outliers with the global estimate
	for (int y = 0; y < g.Height; y++) {
		for (int x = 0; x < g.Width; x++) {
			Vec2f pp = g.At(x, y);
			float d  = g.At(x, y).distance(globalEstimate);
			if (d > maxGlobalDistanceHard) {
				nrep++;
				hasMassiveOutliers = true;
				g.At(x, y)         = globalEstimate;
				gnew.At(x, y)      = globalEstimate;
			}
		}
	}

	float samplesX[filterSizeSQ];
	float samplesY[filterSizeSQ];
	float samplesCleanX[filterSizeSQ];
	float samplesCleanY[filterSizeSQ];
	for (int y = 0; y < g.Height; y++) {
		for (int x = 0; x < g.Width; x++) {
			int i = 0;
			for (int yf = y - filterRadius; yf <= y + filterRadius; yf++) {
				if (yf < 0)
					continue;
				if (yf >= g.Height)
					break;
				for (int xf = x - filterRadius; xf <= x + filterRadius; xf++) {
					if (xf < 0)
						continue;
					if (xf >= g.Width)
						break;
					samplesX[i] = g.At(xf, yf).x;
					samplesY[i] = g.At(xf, yf).y;
					i++;
				}
			}
			pdqsort(samplesX, samplesX + i);
			pdqsort(samplesY, samplesY + i);
			if (fabs(g.At(x, y).x - samplesX[i / 2]) > maxDistance ||
			    fabs(g.At(x, y).y - samplesY[i / 2]) > maxDistance ||
			    fabs(g.At(x, y).x - globalEstimate.x) > maxGlobalDistanceSoft ||
			    fabs(g.At(x, y).y - globalEstimate.y) > maxGlobalDistanceSoft) {
				nrep++;
				// Build up a small set of "clean" samples, which are those that are close to the median.
				// We don't want to be filtering based on dirty samples.
				int iClean = 0;
				for (int j = 0; j < i; j++) {
					if (fabs(samplesX[j] - samplesX[i / 2]) <= maxDistance &&
					    fabs(samplesY[j] - samplesY[i / 2]) <= maxDistance &&
					    fabs(samplesX[j] - globalEstimate.x) <= maxGlobalDistanceSoft &&
					    fabs(samplesY[j] - globalEstimate.y) <= maxGlobalDistanceSoft) {
						samplesCleanX[iClean] = samplesX[j];
						samplesCleanY[iClean] = samplesY[j];
						iClean++;
					}
				}
				if (iClean >= 2) {
					gnew.At(x, y).x = samplesCleanX[iClean / 2];
					gnew.At(x, y).y = samplesCleanY[iClean / 2];
				}
			}
		}
	}
	if (nrep != 0)
		g = gnew;
	return nrep;
}

static void BlurInvalid(DeltaGrid& g, int passes) {
	DeltaGrid   tmp    = g;
	const int   kernel = 1;
	const int   size   = 2 * kernel + 1;
	const float scale  = 1.0f / (float) (size * size);

	DeltaGrid* g1 = &g;
	DeltaGrid* g2 = &tmp;
	for (int pass = 0; pass < passes; pass++) {
		for (int y = kernel; y < g1->Height - kernel; y++) {
			for (int x = kernel; x < g1->Width - kernel; x++) {
				if (g1->IsValid(x, y))
					continue;
				Vec2f avg(0, 0);
				for (int dy = -kernel; dy <= kernel; dy++) {
					for (int dx = -kernel; dx <= kernel; dx++) {
						avg += scale * g1->At(x + dx, y + dy);
					}
				}
				g2->At(x, y) = avg;
			}
		}
		std::swap(g1, g2);
	}
	if (passes % 2 == 1)
		g = tmp;
}

// This thing exists solely to bring sanity to the 4 dimensional array accesses that we do here
struct WarpScore {
	int*     Sum    = nullptr;
	Point32* Result = nullptr;
	int      MW     = 0; // Mesh width
	int      MH     = 0; // Mesh height
	int      MDX    = 0; // Maximum delta x
	int      MDY    = 0; // Maximum delta y
	int      StrideLev1; // The data for a single mesh vertex
	int      StrideLev2; // The data for a row of mesh vertices
	WarpScore(int w, int h, int mdx, int mdy) {
		MW         = w;
		MH         = h;
		MDX        = mdx;
		MDY        = mdy;
		StrideLev1 = MDX * MDY;
		StrideLev2 = StrideLev1 * MW;
		Sum        = new int[h * StrideLev2]();
		Result     = new Point32[MW * MH];
	}
	~WarpScore() {
		delete[] Sum;
		delete[] Result;
	}
	int& At(int mx, int my, int dx, int dy) {
		return Sum[my * StrideLev2 + mx * StrideLev1 + dy * MDX + dx];
	}
	Point32& ResultAt(int mx, int my) {
		return Result[my * MW + mx];
	}

	void BestGlobalDelta(int& dx, int& dy) {
		int bestSum = INT32_MAX;
		for (int y = 0; y < MDY; y++) {
			for (int x = 0; x < MDX; x++) {
				int sum = 0;
				for (int my = 0; my < MH; my++) {
					for (int mx = 0; mx < MW; mx++) {
						sum += At(mx, my, x, y);
					}
				}
				if (sum < bestSum) {
					bestSum = sum;
					dx      = x;
					dy      = y;
				}
			}
		}
	}

	void BruteForceBestFit(int maxHDiv, int maxVDiv) {
		//typedef pair<int, int> ScoreAndIndex;
		//vector<ScoreAndIndex>  scores;
		//scores.resize(MDX * MDY);

		int     bestTotalDx  = 0;
		int     bestTotalDy  = 0;
		int64_t bestTotalSum = INT64_MAX;

		for (int dy = 0; dy < MDY; dy++) {
			for (int dx = 0; dx < MDX; dx++) {
				int64_t totalSum = 0;
				for (int my = 0; my < MH; my++) {
					for (int mx = 0; mx < MW; mx++) {
						int     bestDx;
						int     bestDy;
						int64_t bestSum;
						BestDeltaWithinDivergenceLimit(mx, my, dx, dy, maxHDiv, maxVDiv, bestDx, bestDy, bestSum);
						if (bestSum < 0) {
							tsf::print("negative\n");
							IMQS_ASSERT(false);
						}
						totalSum += bestSum;
					}
				}
				if (totalSum < bestTotalSum) {
					bestTotalSum = totalSum;
					bestTotalDx  = dx;
					bestTotalDy  = dy;
				}
			}
		}

		bool debugPrint = false;
		if (debugPrint)
			tsf::print("---------------- global fit -----------------------------------------------------------------------------\n");

		for (int my = 0; my < MH; my++) {
			for (int mx = 0; mx < MW; mx++) {
				int     bestDx;
				int     bestDy;
				int64_t bestSum;
				BestDeltaWithinDivergenceLimit(mx, my, bestTotalDx, bestTotalDy, maxHDiv, maxVDiv, bestDx, bestDy, bestSum);
				ResultAt(mx, my) = Point32(bestDx, bestDy);
				if (debugPrint)
					tsf::print("%4d ", bestDy);
			}
		}
		if (debugPrint)
			tsf::print("\n");

		/*
		for (int my = 0; my < MH; my++) {
			for (int mx = 0; mx < MW; mx++) {
				// Try the top 5 positions
				for (int dy = 0; dy < MDY; dy++) {
					for (int dx = 0; dx < MDX; dx++) {
						int index     = dx + dy * MDX;
						scores[index] = ScoreAndIndex(At(mx, my, dx, dy), index);
					}
				}
				pdqsort(scores.begin(), scores.end());
				for (int i = 0; i < 15; i++)
					tsf::print("%v,%v\n", scores[i].first, scores[i].second);

				// Try all other positions that are within legal distance from this dx,dy
				int topn = 5;
				for (int i = 0; i < topn; i++) {
				}
			}
		}
		*/
	}

	void BestDeltaWithinDivergenceLimit(int mx, int my, int cdx, int cdy, int maxDivH, int maxDivV, int& bestDx, int& bestDy, int64_t& bestSum) {
		int     dxS  = max(cdx - maxDivH, 0);
		int     dxE  = min(cdx + maxDivH, MDX);
		int     dyS  = max(cdy - maxDivV, 0);
		int     dyE  = min(cdy + maxDivV, MDY);
		int64_t best = INT64_MAX;
		for (int dx = dxS; dx < dxE; dx++) {
			for (int dy = dyS; dy < dyE; dy++) {
				int64_t sum = At(mx, my, dx, dy);
				if (sum < best) {
					bestDx = dx;
					bestDy = dy;
					best   = sum;
				}
			}
		}
		bestSum = best;
	}
};

int FixElementsTooFarFromGlobalBest(Mesh& warpMesh, Rect32 warpMeshValidRect, Vec2f bias, Point32 bestDelta) {
	Vec2f bestDeltaF((float) bestDelta.x, (float) bestDelta.y);
	float maxDistSQ = 20 * 20;

	int nfixed = 0;
	for (int cy = warpMeshValidRect.y1; cy < warpMeshValidRect.y2; cy++) {
		for (int cx = warpMeshValidRect.x1; cx < warpMeshValidRect.x2; cx++) {
			Vec2f raw   = warpMesh.At(cx, cy).Pos;
			Vec2f delta = warpMesh.At(cx, cy).Pos - bias;
			if (delta.distance2D(bestDeltaF) > maxDistSQ) {
				nfixed++;
				warpMesh.At(cx, cy).Pos = bias + bestDeltaF;
			}
		}
	}
	return nfixed;
}

// We compute the transformed mesh of warpImg, so that it aligns to stableImg
// All pixels in stableImg are expected to be defined, but we allow blank (zero alpha) pixels
// in warpImg, and we make sure that we don't try to align any grid cells that have
// one or more blank pixels inside them.
FlowResult OpticalFlow::Frame(Mesh& warpMesh, Frustum warpFrustum, const gfx::Image& _warpImg, const gfx::Image& _stableImg, gfx::Vec2f& bias) {
	FlowResult result;
	int        frameNumber = HistorySize++;

	bool  haveFrustum = warpFrustum.Width != 0;
	Vec2f warpFrustumPoly[4];

	if (haveFrustum) {
		IMQS_ASSERT(_warpImg.Width == warpFrustum.Width); // just sanity check
		// subtle things:
		// shrink X by 1, to ensure we don't touch any black pixels outside the frustum
		// expand Y by 0.1, so that our bottom-most vertices, which are built to butt up
		// against the bottom, are considered inside.
		warpFrustum.Polygon(warpFrustumPoly, -1, 0.1);
	}

	bool drawDebugImages   = false;
	bool debugMedianFilter = false;

	Image  warpImgCopy;
	Image  stableImgCopy;
	Image* warpImg   = nullptr;
	Image* stableImg = nullptr;

	if (!UseRGB) {
		warpImgCopy   = _warpImg.AsType(ImageFormat::Gray);
		stableImgCopy = _stableImg.AsType(ImageFormat::Gray);
		warpImg       = &warpImgCopy;
		stableImg     = &stableImgCopy;
	} else {
		warpImgCopy   = _warpImg;
		stableImgCopy = _stableImg;
		warpImg       = &warpImgCopy;
		stableImg     = &stableImgCopy;
	}

	int minHSearch = -StableHSearchRange;
	int maxHSearch = StableHSearchRange;
	int minVSearch = -StableVSearchRange;
	int maxVSearch = StableVSearchRange;

	// Compute initial bias, which is the (guessed) vector that takes the top-left corner of warpMesh,
	// and moves it to the top-left corner of stableImg.
	if (frameNumber == 0) {
		// warpImg is a raw 'flat' image, which is typically large.
		// stableImg is a small extract from the bottom middle of the first flat image.
		// We assume that stableImg lies in the center of warpImg, and at the bottom.
		// Also, we assume zero motion.
		bias.x = (warpImg->Width - stableImg->Width) / 2;
		//bias.y     = warpImg.Height - stableImg.Height + AbsMaxVSearch;
		bias.y     = warpImg->Height - stableImg->Height;
		bias       = -bias; // get it from warp -> stable
		minHSearch = AbsMinHSearch;
		maxHSearch = AbsMaxHSearch;
		minVSearch = AbsMinVSearch;
		maxVSearch = AbsMaxVSearch;
	}

	// We're aligning on whole pixels here
	bias.x = floor(bias.x + 0.5f);
	bias.y = floor(bias.y + 0.5f);

	// apply the bias to the entire mesh
	for (int i = 0; i < warpMesh.Count; i++)
		warpMesh.Vertices[i].Pos += bias;

	// Decide which grid cells to attempt alignment on
	for (int y = 0; y < warpMesh.Height; y++) {
		for (int x = 0; x < warpMesh.Width; x++) {
			Vec2f p  = warpMesh.At(x, y).Pos;
			Vec2f p1 = p - Vec2f(MatchRadius, MatchRadius) + Vec2f(minHSearch, minVSearch);
			Vec2f p2 = p + Vec2f(MatchRadius, MatchRadius) + Vec2f(maxHSearch, maxVSearch);
			if (p1.x < 0 || p1.y < 0 || p2.x > (float) stableImg->Width || p2.y > (float) stableImg->Height) {
				// search for matching cell would reach outside of stable image
				warpMesh.At(x, y).IsValid = false;
				continue;
			}
			p  = warpMesh.At(x, y).UV;
			p1 = p - Vec2f(MatchRadius, MatchRadius);
			p2 = p + Vec2f(MatchRadius, MatchRadius);
			if (p1.x < 0 || p1.y < 0 || p2.x > (float) warpImg->Width || p2.y > (float) warpImg->Height) {
				// search for matching cell would reach outside of warp image
				warpMesh.At(x, y).IsValid = false;
				continue;
			}
			if (haveFrustum) {
				if (!geom2d::PtInsidePoly(p1.x, p1.y, 4, &warpFrustumPoly[0].x, 2) || !geom2d::PtInsidePoly(p2.x, p2.y, 4, &warpFrustumPoly[0].x, 2)) {
					// warp cell has pixels that are outside of the frustum (ie inside the triangular black regions on the bottom left/right of the flattened image)
					warpMesh.At(x, y).IsValid = false;
					continue;
				}
			}
		}
	}

	// compute a box around all of the warp grid cells that are valid
	vector<Point32> validCells;
	Rect32          warpMeshValidRect  = Rect32::Inverted();
	Rect32          warpImgValidRect   = Rect32::Inverted();
	Rect32          stableImgValidRect = Rect32::Inverted();
	for (int y = 0; y < warpMesh.Height; y++) {
		for (int x = 0; x < warpMesh.Width; x++) {
			if (warpMesh.At(x, y).IsValid) {
				validCells.emplace_back(x, y);
				warpMeshValidRect.ExpandToFit(x, y);
				auto uv  = warpMesh.At(x, y).UV;
				auto pos = warpMesh.At(x, y).Pos;
				warpImgValidRect.ExpandToFit(uv.x - MatchRadius, uv.y - MatchRadius);
				warpImgValidRect.ExpandToFit(uv.x + MatchRadius, uv.y + MatchRadius);
				stableImgValidRect.ExpandToFit(pos.x - MatchRadius + minHSearch, pos.y - MatchRadius + minVSearch);
				stableImgValidRect.ExpandToFit(pos.x + MatchRadius + maxHSearch, pos.y + MatchRadius + maxVSearch);
			}
		}
	}
	warpMeshValidRect.x2++;
	warpMeshValidRect.y2++;
	IMQS_ASSERT(warpMeshValidRect.x1 >= 0);
	IMQS_ASSERT(warpMeshValidRect.y1 >= 0);
	IMQS_ASSERT(warpMeshValidRect.x2 <= warpMesh.Width);
	IMQS_ASSERT(warpMeshValidRect.y2 <= warpMesh.Height);

	//warpMesh.PrintValid();

	// It is pointless applying our transformations to areas of the image that won't be read, so we ensure
	// that we're only doing this on relevant parts of the image.
	// Because we're doing a blur, we make sure that we buffer the rectangle a little.
	auto stableRectBuffer = stableImgValidRect;
	auto warpRectBuffer   = warpImgValidRect;
	stableRectBuffer.Expand(10, 10);
	warpRectBuffer.Expand(10, 10);
	stableRectBuffer.CropTo(Rect32(0, 0, stableImg->Width, stableImg->Height));
	warpRectBuffer.CropTo(Rect32(0, 0, warpImg->Width, warpImg->Height));
	auto stableImgValid = stableImg->Window(stableRectBuffer);
	auto warpImgValid   = warpImg->Window(warpRectBuffer);
	LocalContrast(warpImgValid, 1, 5);
	LocalContrast(stableImgValid, 1, 5);

	if (drawDebugImages) {
		_warpImg.SaveFile("imgRawWarp.png");
		_stableImg.SaveFile("imgRawStable.png");
		warpImg->SaveFile("imgModWarp.png");
		stableImg->SaveFile("imgModStable.png");
		warpImgValid.SaveFile("imgValidWarp.png");
		stableImgValid.SaveFile("imgValidStable.png");
	}

	// Run multiple passes, where at the end of each pass, we perform maximum likelihood filtering, and then
	// on the subsequent pass, restrict the search window to a much smaller region.

	if (debugMedianFilter)
		tsf::print("------------------------------------------------------------------------------------\n");

	//warpMesh.PrintDeltaPos(warpMeshValidRect, bias);

	// This seems to be a bad idea
	bool doGlobalFit  = false;
	bool useGlobalFit = true;
	if (doGlobalFit) {
		int       dyMin = minVSearch;
		int       dyMax = maxVSearch;
		int       dxMin = minHSearch;
		int       dxMax = maxHSearch;
		int       mW    = warpMeshValidRect.Width();
		int       mH    = warpMeshValidRect.Height();
		WarpScore scores(mW, mH, 1 + dxMax - dxMin, 1 + dyMax - dyMin);
		for (auto& c : validCells) {
			Vec2f  cSrc  = warpMesh.At(c.x, c.y).UV;
			Vec2f  cDst  = warpMesh.At(c.x, c.y).Pos;
			Rect32 rect1 = MakeBoxAroundPoint((int) cSrc.x, (int) cSrc.y, MatchRadius);
			Rect32 rect2 = MakeBoxAroundPoint((int) cDst.x, (int) cDst.y, MatchRadius);
			for (int dy = dyMin; dy <= dyMax; dy++) {
				for (int dx = dxMin; dx <= dxMax; dx++) {
					Rect32 r2 = rect2;
					r2.Offset(dx, dy);
					if (r2.x1 < 0 || r2.y1 < 0 || r2.x2 > stableImg->Width || r2.y2 > stableImg->Height) {
						// skip invalid rectangle which is outside of stableImg
						IMQS_ASSERT(false); // WarpScore brute force algo assumes all delta positions are populated.
						continue;
					}
					int32_t sum = DiffSum(*warpImg, *stableImg, rect1, r2);
					IMQS_ASSERT(sum >= 0);
					scores.At(c.x - warpMeshValidRect.x1, c.y - warpMeshValidRect.y1, dx - dxMin, dy - dyMin) = sum;
				}
			}
		}
		//if (debugMedianFilter)
		//	warpMesh.PrintDeltaPos(warpMeshValidRect, bias);

		/*
		Point32 bestGlobal;
		scores.BestGlobalDelta(bestGlobal.x, bestGlobal.y);
		bestGlobal.x += dxMin;
		bestGlobal.y += dyMin;
		if (useGlobalFit) {
			Vec2f bestGlobalF((float) bestGlobal.x, (float) bestGlobal.y);
			for (auto& c : validCells) {
				Vec2f raw = warpMesh.At(c.x, c.y).Pos;
				warpMesh.At(c.x, c.y).Pos += bestGlobalF;
			}
			minVSearch = max(minVSearch, -20);
			maxVSearch = min(maxVSearch, 20);
			minHSearch = max(minHSearch, -20);
			maxHSearch = min(maxHSearch, 20);
		}
		tsf::print("Best Global: %v, %v\n", bestGlobal.x, bestGlobal.y);
		*/

		if (useGlobalFit) {
			// actual divergence is 2x what we give here, because it is searched for left and right
			scores.BruteForceBestFit(2, 2);

			for (int y = 0; y < mH; y++) {
				for (int x = 0; x < mW; x++) {
					Point32 delta = scores.ResultAt(x, y) + Point32(dxMin, dyMin);
					warpMesh.At(x + warpMeshValidRect.x1, y + warpMeshValidRect.y1).Pos += Vec2f(delta.x, delta.y);
				}
			}

			minVSearch = max(minVSearch, -1);
			maxVSearch = min(maxVSearch, 1);
			minHSearch = max(minHSearch, -1);
			maxHSearch = min(maxHSearch, 1);

			if (debugMedianFilter)
				warpMesh.PrintDeltaPos(warpMeshValidRect, bias);
		}
		if (drawDebugImages) {
			DrawMesh("mesh-prefilter-0.png", *stableImg, warpMesh, true);
			DrawMesh("mesh-prefilter-1.png", *warpImg, warpMesh, false);
		}
	}

	bool hasMassiveOutliers = true;
	int  maxPass            = 2;
	for (int pass = 0; pass < maxPass; pass++) {
		//tsf::print("Flow pass %v\n", pass);
		int dyMin = minVSearch;
		int dyMax = maxVSearch;
		int dxMin = minHSearch;
		int dxMax = maxHSearch;
		if (pass == 1) {
			// refinement after filtering
			//tsf::print("Fine adjustment pass\n");
			int fineAdjust = hasMassiveOutliers ? 8 : 2;
			dyMin          = -fineAdjust;
			dyMax          = fineAdjust;
			dxMin          = -fineAdjust;
			dxMax          = fineAdjust;
		}
		int     searchWindowSize = (dxMax - dxMin) * (dyMax - dyMin);
		int64_t allDiffSum       = 0;
		int     nValidCells      = validCells.size();
		// omp parallel here takes us from 22 milliseconds to 6 milliseconds
		//auto start = time::PerformanceCounter();
#pragma omp parallel for
		for (int iCell = 0; iCell < nValidCells; iCell++) {
			auto& c = validCells[iCell];
			//Vec2f  cSrc    = warpMesh.UVimg(warpImg.Width, warpImg.Height, c.x, c.y);
			Vec2f  cSrc    = warpMesh.At(c.x, c.y).UV;
			Vec2f  cDst    = warpMesh.At(c.x, c.y).Pos;
			Rect32 rect1   = MakeBoxAroundPoint((int) cSrc.x, (int) cSrc.y, MatchRadius);
			Rect32 rect2   = MakeBoxAroundPoint((int) cDst.x, (int) cDst.y, MatchRadius);
			int    bestSum = INT32_MAX;
			int    bestDx  = 0;
			int    bestDy  = 0;
			//int64_t avgSum  = 0;
			for (int dy = dyMin; dy <= dyMax; dy++) {
				for (int dx = dxMin; dx <= dxMax; dx++) {
					Rect32 r2 = rect2;
					r2.Offset(dx, dy);
					if (r2.x1 < 0 || r2.y1 < 0 || r2.x2 > stableImg->Width || r2.y2 > stableImg->Height) {
						// skip invalid rectangle which is outside of stableImg
						continue;
					}
					int32_t sum = DiffSum(*warpImg, *stableImg, rect1, r2);
					//avgSum += sum;
					if (sum < bestSum) {
						bestSum = sum;
						bestDx  = dx;
						bestDy  = dy;
					}
				}
			}
#pragma omp atomic
			allDiffSum += bestSum;
			// I thought this would work well, indicating patches that have good detail for matching, but it doesn't work. No idea why not.
			//warpMesh.At(c.x, c.y).DeltaStrength = float((double) avgSum / (double) searchWindowSize) / ((float) bestSum + 0.1f);
			warpMesh.At(c.x, c.y).Pos += Vec2f(bestDx, bestDy);
		}
		//auto duration = time::PerformanceCounter() - start;
		//tsf::print("flow time: %v microseconds\n", duration / 1000);
		if (debugMedianFilter) {
			warpMesh.PrintDeltaPos(warpMeshValidRect, bias);
			//warpMesh.PrintDeltaStrength(warpMeshValidRect);
		}

		result.Diff = float((double) allDiffSum / (double) validCells.size());

		//if (debugMedianFilter)
		//	warpMesh.PrintDeltaPos(warpMeshValidRect, bias);
		if (pass == 0 && drawDebugImages) {
			DrawMesh("mesh-prefilter-0.png", *stableImg, warpMesh, true);
			DrawMesh("mesh-prefilter-1.png", *warpImg, warpMesh, false);
		}

		int       maxFilterPasses = EnableMedianFilter ? 10 : 0;
		int       nfilterPasses   = 0;
		DeltaGrid dg;
		CopyMeshToDelta(warpMesh, warpMeshValidRect, dg, bias);
		for (int ifilter = 0; ifilter < maxFilterPasses; ifilter++) {
			int nrep = MedianFilter(pass, dg, hasMassiveOutliers);
			if (debugMedianFilter)
				tsf::print("Median Filter replaced %v samples\n", nrep);
			if (nrep == 0)
				break;
			nfilterPasses++;
		}
		CopyDeltaToMesh(dg, warpMesh, warpMeshValidRect, bias);
		if (nfilterPasses == 0) {
			// If we performed no filtering, then a second alignment pass will not change anything
			break;
		}
		if (debugMedianFilter)
			warpMesh.PrintDeltaPos(warpMeshValidRect, bias);
	}

	//UpdateHistoryMesh(warpMesh);

	if (drawDebugImages) {
		DrawMesh("mesh-postfilter-0.png", *stableImg, warpMesh, true);
		DrawMesh("mesh-postfilter-1.png", *warpImg, warpMesh, false);
	}
	//warpMesh.DrawFlowImage(warpMeshValidRect, "flow-diagram.png");

	if (ExtrapolateInvalidCells) {
		// Fill the remaining invalid cells

		// Start by setting all invalid cells to the average displacement
		Vec2f avgDisp(0, 0);
		float avgScale = 1.0f / (float) validCells.size();
		for (auto& c : validCells)
			avgDisp += avgScale * (warpMesh.At(c.x, c.y).Pos - warpMesh.At(c.x, c.y).UV);

		vector<Point32> remain;
		for (int y = 0; y < warpMesh.Height; y++) {
			for (int x = 0; x < warpMesh.Width; x++) {
				if (!warpMesh.At(x, y).IsValid) {
					warpMesh.At(x, y).Pos = warpMesh.At(x, y).UV + avgDisp;
					remain.emplace_back(x, y);
				}
			}
		}

		// For cells higher up, make their horizontal drift = 0, so that we force the system to always move forward in a straight line
		// UPDATE: make them homogenous in X and Y
		if (true) {
			//warpMesh.DrawFlowImage("flow-diagram-all.png");
			//for (int y = 0; y < warpMeshValidRect.y1; y++) {
			//	for (int x = 0; x < warpMesh.Width; x++) {
			//		warpMesh.At(x, y).Pos.x = warpMesh.At(x, y).UV.x + bias.x;
			//	}
			//}
			// This limit here.. the "-4", is intimately tied to the mesh rect that we stitch inside Stitcher2,
			// right before it calls Rend.DrawMesh().
			for (int y = 0; y < warpMesh.Height - 5; y++) {
				for (int x = 0; x < warpMesh.Width; x++) {
					//warpMesh.At(x, y).Pos.x   = warpMesh.At(x, y).UV.x + bias.x; // lock horizontal drift (bad hack)
					warpMesh.At(x, y).Pos.x   = warpMesh.At(x, y).UV.x + avgDisp.x;
					warpMesh.At(x, y).Pos.y   = warpMesh.At(x, y).UV.y + avgDisp.y;
					warpMesh.At(x, y).IsValid = false;
				}
			}
			//warpMesh.DrawFlowImage("flow-diagram-all.png");
		}

		// smooth the invalid cells (but not too high up, since they aren't aligned at all, so that's just computation wasted)
		{
			DeltaGrid dg;
			int       y1 = warpMesh.Height / 2;
			CopyMeshToDelta(warpMesh, Rect32(0, y1, warpMesh.Width, warpMesh.Height), dg, bias);
			BlurInvalid(dg, 3);
			CopyDeltaToMesh(dg, warpMesh, Rect32(0, y1, warpMesh.Width, warpMesh.Height), bias);
			//warpMesh.DrawFlowImage("flow-diagram-all.png");
		}

		// For the cells at the bottom, clone from vertex above
		for (int x = 0; x < warpMesh.Width; x++) {
			auto delta                              = warpMesh.At(x, warpMesh.Height - 2).Pos - warpMesh.At(x, warpMesh.Height - 2).UV;
			warpMesh.At(x, warpMesh.Height - 1).Pos = warpMesh.At(x, warpMesh.Height - 1).UV + delta;
		}

		// lower the opacity of the invalid cells on the bottom
		for (int y = warpMesh.Height - 2; y < warpMesh.Height; y++) {
			for (int x = 0; x < warpMesh.Width; x++) {
				if (!warpMesh.At(x, y).IsValid)
					warpMesh.At(x, y).Color.a = 0;
			}
		}

		// lower the opacity of ALL cells on the bottom
		// Why? Because the outer edges of the frame are darker, because of the cheap lens
		//for (int y = warpMesh.Height - 2; y < warpMesh.Height; y++) {
		//	for (int x = 0; x < warpMesh.Width; x++) {
		//		warpMesh.At(x, y).Color.a = 0;
		//	}
		//}

		// lower the opacity of all cells outside of the frustum
		// NOTE: This is a poor substitute for aligning the outer triangular edges
		// via optical flow. Right now I want to avoid doing that alignment, because
		// of the expensive cost of reading a MUCH larger portion of the framebuffer
		// on which to perform alignment. I will be even more expensive once we do rotation.
		if (haveFrustum) {
			for (int y = 0; y < warpMesh.Height; y++) {
				for (int x = 0; x < warpMesh.Width; x++) {
					Vec2f p = warpMesh.At(x, y).UV;
					// a larger buffer generally improves the quality of the stitching, in the absence of doing any optical flow
					// on the periphery.
					int buffer = 200;
					if (!geom2d::PtInsidePoly(p.x - buffer, p.y, 4, &warpFrustumPoly[0].x, 2) || !geom2d::PtInsidePoly(p.x + buffer, p.y, 4, &warpFrustumPoly[0].x, 2))
						warpMesh.At(x, y).Color.a = 0;
				}
			}
		}

		//warpMesh.DrawFlowImage(tsf::fmt("flow-diagram-all-%d.png", frameNumber));
		//warpMesh.PrintSample(warpMesh.Width / 2, warpMesh.Height - 1);
	}

	return result;
}

void OpticalFlow::DrawMesh(std::string filename, const gfx::Image& img, const Mesh& mesh, bool isStable) {
	Canvas c(img.Width, img.Height);
	c.GetImage()->CopyFrom(img);
	for (int y = 0; y < mesh.Height; y++) {
		for (int x = 0; x < mesh.Width; x++) {
			auto& v = mesh.At(x, y);
			if (!v.IsValid)
				continue;
			RectF box = RectF::Inverted();
			if (isStable)
				box.ExpandToFit(v.Pos.x, v.Pos.y);
			else
				box.ExpandToFit(v.UV.x, v.UV.y);
			box.Expand(MatchRadius, MatchRadius);
			c.Rect(box, Color8(255, 0, 0, 255), 1.0);
		}
	}
	c.GetImage()->SaveFile(filename);
}

// Compute sum of absolute differences
static int32_t DiffSum(const Image& img1, const Image& img2, Rect32 rect1, Rect32 rect2) {
	IMQS_ASSERT(img1.Format == img2.Format);
	IMQS_ASSERT(img1.NumChannels() == 1 || img1.NumChannels() == 4);
	IMQS_ASSERT(rect1.Width() == rect2.Width());
	IMQS_ASSERT(rect1.Height() == rect2.Height());
	int     w   = rect1.Width();
	int     h   = rect1.Height();
	int32_t sum = 0;
	for (int y = 0; y < h; y++) {
		if (img1.NumChannels() == 1) {
			const uint8_t* p1 = img1.At(rect1.x1, rect1.y1 + y);
			const uint8_t* p2 = img2.At(rect2.x1, rect2.y1 + y);
			for (int x = 0; x < w; x++) {
				int d = (int) p1[x] - (int) p2[x];
				sum += abs(d);
			}
		} else {
			const Color8* p1 = (const Color8*) img1.At(rect1.x1, rect1.y1 + y);
			const Color8* p2 = (const Color8*) img2.At(rect2.x1, rect2.y1 + y);
			for (int x = 0; x < w; x++) {
				int r = abs((int) p1[x].r - (int) p2[x].r);
				int g = abs((int) p1[x].g - (int) p2[x].g);
				int b = abs((int) p1[x].b - (int) p2[x].b);
				sum += r + g + b;
			}
		}
	}
	return sum;
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

static void LocalContrast(Image& img, int size, int iterations) {
	Image blur = img;
	blur.BoxBlur(size, iterations);
	//blur.SaveFile("blur.png");
	for (int y = 0; y < img.Height; y++) {
		uint8_t* src = blur.Line(y);
		uint8_t* dst = img.Line(y);
		if (img.NumChannels() == 1) {
			for (int x = 0; x < img.Width; x++) {
				int diff = (int) *dst - (int) *src;
				*dst     = (uint8_t) math::Clamp<int>(diff * DebugBrightenLocalContrast + 127, 0, 255);
				src++;
				dst++;
			}
		} else {
			IMQS_ASSERT(img.NumChannels() == 4);
			for (int x = 0; x < img.Width; x++) {
				int diffR = (int) dst[0] - (int) src[0];
				int diffG = (int) dst[1] - (int) src[1];
				int diffB = (int) dst[2] - (int) src[2];
				diffR     = math::Clamp<int>(diffR * DebugBrightenLocalContrast + 127, 0, 255);
				diffG     = math::Clamp<int>(diffG * DebugBrightenLocalContrast + 127, 0, 255);
				diffB     = math::Clamp<int>(diffB * DebugBrightenLocalContrast + 127, 0, 255);
				dst[0]    = (uint8_t) diffR;
				dst[1]    = (uint8_t) diffG;
				dst[2]    = (uint8_t) diffB;
				src += 4;
				dst += 4;
			}
		}
	}
}

} // namespace roadproc
} // namespace imqs