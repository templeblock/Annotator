#pragma once

#include "Mesh.h"
#include "Perspective.h"

namespace imqs {
namespace roadproc {

int64_t DiffSum(const gfx::Image& img1, const gfx::Image& img2, gfx::Rect32 rect1, gfx::Rect32 rect2);
void    LocalContrast(gfx::Image& img, int size, int iterations);

struct DeltaGrid {
	int               Width  = 0;
	int               Height = 0;
	gfx::Vec2f*       Delta  = nullptr;
	bool*             Valid  = nullptr;
	gfx::Vec2f&       At(int x, int y) { return Delta[y * Width + x]; }
	const gfx::Vec2f& At(int x, int y) const { return Delta[y * Width + x]; }
	bool&             IsValid(int x, int y) { return Valid[y * Width + x]; }
	bool              IsValid(int x, int y) const { return Valid[y * Width + x]; }

	DeltaGrid();
	DeltaGrid(DeltaGrid& g);
	~DeltaGrid();

	void Alloc(int w, int h);

	DeltaGrid& operator=(const DeltaGrid& g);
};

struct FlowResult {
	float Diff = 0;
};

class OpticalFlow {
public:
	int GridW       = 0;
	int GridH       = 0;
	int MatchRadius = 12; // we match a square of MatchRadius x MatchRadius pixels,

	int AbsMinHSearch = 0; // minimum horizontal displacement for alignment points
	int AbsMaxHSearch = 0; // maximum horizontal displacement for alignment points

	int AbsMinVSearch = 0; // minimum vertical displacement for alignment points (driving forwards)
	int AbsMaxVSearch = 0; // maximum vertical displacement for alignment points (driving backwards)

	int StableHSearchRange = 0; // Max horizontal diversion, frame to frame - needs to be more than V, because of camera pointing left/right away from straight ahead
	int StableVSearchRange = 0; // Max vertical diversion, frame to frame

	bool UseRGB                  = true;  // If false, then convert images to gray before performing optical flow
	bool ExtrapolateInvalidCells = false; // If true, then extrapolate valid cells to all of the other cells which were not aligned
	bool EnableMedianFilter      = true;

	OpticalFlow();

	void SetupSearchDistances(int rawVideoWidth);

	FlowResult Frame(Mesh& warpMesh, Frustum warpFrustum, const gfx::Image& warpImg, const gfx::Image& stableImg, gfx::Vec2f& bias);

private:
	int HistorySize = 0;
	//Mesh HistoryMesh;

	void DrawMesh(std::string filename, const gfx::Image& img, const Mesh& mesh, bool isStable);
};

} // namespace roadproc
} // namespace imqs