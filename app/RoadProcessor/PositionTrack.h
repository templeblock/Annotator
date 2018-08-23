#pragma once

#include "Proj.h"

namespace imqs {
namespace roadproc {

struct TimePos {
	double     FrameTime;
	gfx::Vec3d Pos;
};

// A time vs position track. Position comes from GPS.
class PositionTrack {
public:
	bool                 IsProjected = false; // If true, then position values are Web Mercator meters
	std::vector<TimePos> Values;

	PositionTrack();
	~PositionTrack();

	Error      LoadFile(std::string filename);
	Error      SaveFile(std::string filename);
	Error      SaveCSV(std::string filename);
	bool       GetPositionAndVelocity(double time, gfx::Vec3d& pos, gfx::Vec2d& vel2D) const; // Returns position in web mercator meters, velocity in meters/second
	gfx::Vec3d GetPosition(double time) const;                                                // Returns position in web mercator meters
	void       Dump(double start, double end, double interval) const;
	void       DumpRaw(size_t start, size_t end) const;
	void       ConvertToWebMercator();

	// Simplify the positions with Douglas Puecker
	void Simplify(double tolerance);

	// Smooth the positions by generating cubic bezier curves that run through all of the existing points,
	// and replaces 'Values' with those new points.
	// The new values are approximately 'increment' meters apart.
	// 'smoothness' is a value from 0 to 1, where 0 is just straight lines, and 1 is very curvy.
	void Smooth(double smoothness, double increment);

private:
	Proj OutProj;
};

} // namespace roadproc
} // namespace imqs