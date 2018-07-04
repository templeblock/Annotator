#pragma once

namespace imqs {
namespace roadproc {

class OpticalFlow {
public:
	OpticalFlow();

	void Frame(gfx::Image& img1, gfx::Image& img2);

private:
	int                       CellSize           = 16; // Match cells of CellSize x CellSize pixels
	int                       GridW              = 0;
	int                       GridH              = 0;
	double                    LastDx             = 0;
	double                    LastDy             = 0;
	int                       HistorySize        = 0;
	int                       AbsMinHSearch      = -20; // minimum horizontal displacement for alignment points
	int                       AbsMaxHSearch      = 20;  // maximum horizontal displacement for alignment points
	int                       AbsMinVSearch      = -20; // minimum horizontal displacement for alignment points (driving backwards)
	int                       AbsMaxVSearch      = 300; // maximum horizontal displacement for alignment points (driving forwards)
	int                       StableHSearchRange = 20;  // Max diversion, frame to frame
	int                       StableVSearchRange = 20;  // Max diversion, frame to frame
	std::vector<gfx::Point32> GridCenter;               // Center of grid cell
	std::vector<gfx::Vec2d>   LastGrid;                 // last grid delta

	gfx::Point32& GridCenterAt(int x, int y) { return GridCenter[y * GridW + x]; }
	gfx::Vec2d&   GridEl(std::vector<gfx::Vec2d>& grid, int x, int y) { return grid[y * GridW + x]; }
	gfx::Vec2d&   LastGridEl(int x, int y) { return LastGrid[y * GridW + x]; }

	void SetupGrid(gfx::Image& img1);
	void PrintGrid(int dim);
};

} // namespace roadproc
} // namespace imqs