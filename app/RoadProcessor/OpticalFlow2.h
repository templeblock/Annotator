#pragma once

#include "Mesh.h"
#include "Perspective.h"

namespace imqs {
namespace roadproc {

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

class OpticalFlow2 {
public:
	int GridW              = 0;
	int GridH              = 0;
	int MatchRadius        = 0;
	int AbsMinHSearch      = -10;  // minimum horizontal displacement for alignment points
	int AbsMaxHSearch      = 10;   // maximum horizontal displacement for alignment points
	int AbsMinVSearch      = -200; // minimum horizontal displacement for alignment points (driving forwards)
	int AbsMaxVSearch      = 20;   // maximum horizontal displacement for alignment points (driving backwards)
	int StableHSearchRange = 10;   // Max diversion, frame to frame
	int StableVSearchRange = 20;   // Max diversion, frame to frame

	OpticalFlow2();

	void Frame(Mesh& warpMesh, Frustum warpFrustum, gfx::Image& warpImg, gfx::Image& stableImg, gfx::Vec2f& bias);

	gfx::Point32& GridCenterAt(int x, int y) { return GridCenter[y * GridW + x]; }
	gfx::Vec2d&   LastGridEl(int x, int y) { return LastGrid[y * GridW + x]; }

private:
	int                       CellSize    = 16; // Match cells of CellSize x CellSize pixels
	double                    LastDx      = 0;
	double                    LastDy      = 0;
	int                       HistorySize = 0;
	std::vector<gfx::Point32> GridCenter; // Center of grid cell
	std::vector<gfx::Vec2d>   LastGrid;   // last grid delta

	gfx::Vec2d& GridEl(std::vector<gfx::Vec2d>& grid, int x, int y) { return grid[y * GridW + x]; }

	//void SetupGrid(gfx::Image& img1);
	void PrintGrid(int dim);
	void DrawGrid(gfx::Image& img1);
};

} // namespace roadproc
} // namespace imqs