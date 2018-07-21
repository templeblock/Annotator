#include "pch.h"
#include "PositionTrack.h"
#include "Globals.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

static int FindTime(const TimePos& tp, const double& time) {
	return math::SignOrZero(tp.Time - time);
}

PositionTrack::PositionTrack() {
	OutProj.Init("+proj=longlat +datum=WGS84 +no_defs", "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=20037508.3427892430766 +y_0=-20037508.3427892430766 +k=1.0 +units=m +nadgrids=@null +axis=esu +no_defs +over");
}

PositionTrack::~PositionTrack() {
}

Error PositionTrack::LoadFile(std::string filename) {
	nlohmann::json j;
	auto           err = nj::ParseFile(filename, j);
	if (!err.OK())
		return err;

	for (size_t i = 0; i < j.size(); i++) {
		TimePos tp;
		tp.Time  = j[i]["time"];
		tp.Pos.x = j[i]["lon"];
		tp.Pos.y = j[i]["lat"];
		Values.push_back(tp);
	}

	return Error();
}

bool PositionTrack::GetPositionAndVelocity(double time, gfx::Vec3d& pos, gfx::Vec2d& vel2D) const {
	if (time < 0 || time > Values.back().Time)
		return false;
	size_t i = algo::BinarySearchTry(Values.size(), &Values[0], time, FindTime);
	i--;
	if (i == -1 || i >= Values.size() - 1)
		return false;

	const auto& p1   = Values[i];
	const auto& p2   = Values[i + 1];
	double      frac = (time - p1.Time) / (p2.Time - p1.Time);

	auto pp1 = OutProj.Convert(p1.Pos);
	auto pp2 = OutProj.Convert(p2.Pos);
	//Vec3d pp1 = p1.Pos;
	//Vec3d pp2 = p2.Pos;

	pos            = pp1 + frac * (pp2 - pp1);
	auto   delta2D = (pp2 - pp1).vec2;
	double speed   = delta2D.size() / (p2.Time - p1.Time);
	//vel2D          = 10000.0 * speed * delta2D.normalized();
	vel2D = speed * delta2D.normalized();

	//size_t i0 = i - 1;
	//size_t i1 = i;
	//size_t i2 = i + 1;

	return true;
}

void PositionTrack::Dump(double start, double end, double interval) const {
	for (double t = start; t < end; t += interval) {
		Vec3d pos;
		Vec2d vel;
		GetPositionAndVelocity(t, pos, vel);
		tsf::print("%4.1f, %12.1f %12.1f, %6.3f %6.3f (%6.3f)\n", t, pos.x, pos.y, vel.x, vel.y, vel.size());
	}
}

} // namespace roadproc
} // namespace imqs