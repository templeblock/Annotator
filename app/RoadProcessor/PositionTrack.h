#pragma once

#include "Proj.h"

namespace imqs {
namespace roadproc {

struct TimePos {
	double     Time;
	gfx::Vec3d Pos;
};

// A time vs position track. Position comes from GPS.
class PositionTrack {
public:
	std::vector<TimePos> Values;

	PositionTrack();
	~PositionTrack();
	Error LoadFile(std::string filename);
	bool  GetPositionAndVelocity(double time, gfx::Vec3d& pos, gfx::Vec2d& vel2D) const;
	void  Dump(double start, double end, double interval) const;

private:
	Proj OutProj;
};

} // namespace roadproc
} // namespace imqs