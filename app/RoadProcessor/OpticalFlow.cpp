#include "pch.h"
#include "OpticalFlow.h"

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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

OpticalFlow::OpticalFlow() {
}

void OpticalFlow::SetupGrid(Image& img1) {
	// These windows and regions could be setup in a range of different configurations, and they could
	// be more flexible at runtime. I'm just nailing it down here to make it easier to prototype and think about.
	int vWindow = img1.Height;

	// CHANGE THAT... now img1 is the entire match window. We are going to be blending over the entire img1.

	// We take the bottom 900 pixels of img1, and divide it into 3 equal parts.
	// The bottom part will remain constant.
	// The middle part will be matched to the incoming frame (img2), and blended with it.
	// The top part will come entirely from img2 (or perhaps blended also).

	IMQS_ASSERT(AbsMaxHSearch == -AbsMinHSearch);

	int availWidth  = img1.Width - AbsMaxHSearch * 2; // outer edges of grid cells touch up against this size
	int availHeight = vWindow;                        // outer edges of grid cells touch up against this size
	availWidth -= CellSize;                           // remove 1/2 a cell width from left, and 1/2 a cell width from right
	availHeight -= CellSize;                          // same for vertical

	// Populate the grid centers
	int gridLeft = (img1.Width - availWidth) / 2;
	//int gridTop  = img1.Height - vWindow * 2 + CellSize / 2;
	int gridTop = CellSize / 2;

	GridW = availWidth / (CellSize * 3); // *3 is thumbsuck
	GridH = availHeight / (CellSize * 3);
	GridW = min(GridW, 16);
	GridH = min(GridH, 16);

	GridCenter.resize(GridW * GridH);

	for (int x = 0; x < GridW; x++) {
		int cx = gridLeft + (x * availWidth) / (GridW - 1);
		for (int y = 0; y < GridH; y++) {
			int cy             = gridTop + (y * availHeight) / (GridH - 1);
			GridCenterAt(x, y) = Point32(cx, cy);
		}
	}

	LastGrid.resize(GridW * GridH);
	for (auto& v : LastGrid)
		v = Vec2d(0, 0);
}

// We compute the transform from img1 to img2
// We expect img1 to be a small window, like 1920x300, and img2 to be a larger unprojected camera frame
void OpticalFlow::Frame(Image& img1, Image& img2) {
	int frameNumber = HistorySize++;

	if (frameNumber == 0)
		SetupGrid(img1);

	Image img1G = img1.AsType(ImageFormat::Gray);
	Image img2G = img2.AsType(ImageFormat::Gray);

	const Image& tiny1 = img1G;
	const Image& tiny2 = img2G;

	//tiny1.SavePng("tiny1.png");
	//tiny2.SavePng("tiny2.png");

	int imgScale = 1;

	int minVSearch = AbsMinVSearch + FirstFrameBiasV;
	int maxVSearch = AbsMaxVSearch + FirstFrameBiasV;
	int minHSearch = AbsMinHSearch + FirstFrameBiasH;
	int maxHSearch = AbsMaxHSearch + FirstFrameBiasH;

	vector<double> allDx, allDy;
	vector<Vec2d>  grid;
	vector<double> allD;
	grid.resize(GridW * GridH);
	double avgSD   = 0;
	double avgDiff = 0;
	for (int gx = 0; gx < GridW; gx++) {
		for (int gy = 0; gy < GridH; gy++) {
			if (frameNumber > 0) {
				minVSearch = (int) (LastGridEl(gx, gy).y - StableVSearchRange / (double) imgScale);
				maxVSearch = (int) (LastGridEl(gx, gy).y + StableVSearchRange / (double) imgScale);
				minHSearch = (int) (LastGridEl(gx, gy).x - StableHSearchRange / (double) imgScale);
				maxHSearch = (int) (LastGridEl(gx, gy).x + StableHSearchRange / (double) imgScale);
				minHSearch = max(minHSearch, AbsMinHSearch + FirstFrameBiasH);
				maxHSearch = min(maxHSearch, AbsMaxHSearch + FirstFrameBiasH);
				minVSearch = max(minVSearch, AbsMinVSearch + FirstFrameBiasV);
				maxVSearch = min(maxVSearch, AbsMaxVSearch + FirstFrameBiasV);
			}
			Point32 center = GridCenterAt(gx, gy);
			Rect32  rect(center.x, center.y, center.x, center.y);
			rect.Expand(CellSize / 2, CellSize / 2);
			double startBestD = 1e10;
			double bestD      = startBestD;
			int    bestDx     = 0;
			int    bestDy     = 0;
			allD.clear();
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
			GridEl(grid, gx, gy).x = bestDx;
			GridEl(grid, gx, gy).y = bestDy;
		}
	}

	pdqsort(allDx.begin(), allDx.end());
	pdqsort(allDy.begin(), allDy.end());
	//double finalDx = allDx[allDx.size() / 2];
	//double finalDy = allDy[allDy.size() / 2];
	int    topN    = 15;
	auto   allDxMV = math::MeanAndVariance<double, double>(topN, &allDx[allDx.size() - topN]);
	auto   allDyMV = math::MeanAndVariance<double, double>(topN, &allDy[allDy.size() - topN]);
	double finalDx = allDxMV.first;
	double finalDy = allDyMV.first;
	tsf::print("%.2f,%.2f,%.2f,%.2f,%.2f\n", (double) frameNumber / 29.97, finalDy, finalDx, avgSD, avgDiff);

	// might need to compute max vehicle accel/break speed to know what this number 'a' should be
	double a = frameNumber == 0 ? 1 : 0.2;
	LastDx   = LastDx + a * (finalDx - LastDx);
	LastDy   = LastDy + a * (finalDy - LastDy);
	// Update running averages of grid offsets
	for (int gy = 0; gy < GridH; gy++) {
		for (int gx = 0; gx < GridW; gx++) {
			LastGridEl(gx, gy) += a * (GridEl(grid, gx, gy) - LastGridEl(gx, gy));
			//GridEl(gx, gy).x += a * (GridEl(gx, gy).x - GridEl(gx, gy).x);
			//GridEl(gx, gy).y += a * (GridEl(gx, gy).y - GridEl(gx, gy).y);
		}
	}
	PrintGrid(2);
	//DrawGrid(img1);
}

// 0: x
// 1: y
// 2: x y
void OpticalFlow::PrintGrid(int dim) {
	for (int y = 0; y < GridH; y++) {
		for (int x = 0; x < GridW; x++) {
			if (dim == 0 || dim == 2)
				tsf::print("%3.0f ", LastGridEl(x, y).x);
			else
				tsf::print("%4.0f ", LastGridEl(x, y).y);
		}
		if (dim == 2) {
			tsf::print(" | ");
			for (int x = 0; x < GridW; x++)
				tsf::print("%4.0f ", LastGridEl(x, y).y);
		}
		tsf::print("\n");
	}
}

void OpticalFlow::DrawGrid(Image& img1) {
	Canvas c;
	float  gridSpaceX = (GridCenterAt(GridW - 1, 0).x - GridCenterAt(0, 0).x) / GridW;
	float  gridSpaceY = (GridCenterAt(0, GridH - 1).y - GridCenterAt(0, 0).y) / GridH;
	c.Alloc((GridW + 1.5) * gridSpaceX, (GridH + 1.5) * gridSpaceY, Color8(255, 255, 255, 255));

	Vec2d avgD(0, 0);
	for (int x = 0; x < GridW; x++) {
		for (int y = 0; y < GridH; y++) {
			avgD += LastGridEl(x, y) / (double) (GridW * GridH);
		}
	}

	Point32 topG = GridCenterAt(0, 0);

	for (int x = 0; x < GridW; x++) {
		for (int y = 0; y < GridH; y++) {
			auto p = GridCenterAt(x, y) - topG;
			p.x += (int) (gridSpaceX / 1.5);
			p.y += (int) (gridSpaceY / 1.5);
			auto d = LastGridEl(x, y);
			d.y -= avgD.y;
			c.FillCircle(p.x, p.y, 1.2, Color8(150, 0, 0, 255));
			c.StrokeLine(p.x, p.y, p.x + d.x, p.y + d.y, Color8(150, 0, 0, 255), 1.0f);
		}
	}
	c.GetImage()->SavePng("flow-grid.png");
}

} // namespace roadproc
} // namespace imqs