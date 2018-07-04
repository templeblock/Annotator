#pragma once

namespace imqs {
namespace roadproc {

// This class is a wrapper around lensfun.
// It has been built only to do distortion correction, not any of the other
// things like perspective or color. By restriction ourselves to just distortion,
// we can perform distortion correction and de-perspective in a single step.
class LensCorrector {
public:
	lfDatabase*     DB        = nullptr;
	const lfCamera* Camera    = nullptr;
	const lfLens*   Lens      = nullptr;
	float*          InterpPos = nullptr; // Holds the interpolation coordinates from ComputeDistortionForLine

	LensCorrector();
	~LensCorrector();
	Error LoadDatabase(std::string dbPath);
	Error LoadCameraAndLens(std::string spec);
	Error LoadCameraAndLens(std::string camera, std::string lens);

	Error InitializeDistortionCorrect(int width, int height);

	// After running this function, you can get the coordinates for the XY RGB coordinates out of InterpPos
	void ComputeDistortionForLine(int y);

	Error Correct(gfx::Image& raw, gfx::Image& fixed); // the raw image is destroyed

private:
	lfModifier* Mod        = nullptr;
	int         ImageWidth = 0;
};

} // namespace roadproc
} // namespace imqs