#include "pch.h"
#include "PositionTrack.h"
#include "Globals.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

static int FindTime(const TimePos& tp, const double& frameTime) {
	return math::SignOrZero(tp.FrameTime - frameTime);
}

PositionTrack::PositionTrack() {
	OutProj.Init("+proj=longlat +datum=WGS84 +no_defs",
	             "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=20037508.3427892430766 +y_0=-20037508.3427892430766 +k=1.0 +units=m +nadgrids=@null +axis=esu +no_defs +over");
}

PositionTrack::~PositionTrack() {
}

Error PositionTrack::LoadFile(std::string filename) {
	nlohmann::json j;
	auto           err = nj::ParseFile(filename, j);
	if (!err.OK())
		return err;

	Values.clear();
	IsProjected = false;

	for (size_t i = 0; i < j.size(); i++) {
		TimePos tp;
		// The JSON also stores absolute time, as "time"
		tp.FrameTime = j[i]["dt"];
		tp.Pos.x     = j[i]["lon"];
		tp.Pos.y     = j[i]["lat"];
		tp.Pos.z     = 0;
		Values.push_back(tp);
	}

	return Error();
}

Error PositionTrack::SaveFile(std::string filename) {
	nlohmann::json j;

	for (const auto& v : Values) {
		nlohmann::json el;
		el["dt"]  = v.FrameTime;
		el["lon"] = v.Pos.x;
		el["lat"] = v.Pos.y;
		j.push_back(std::move(el));
	}

	return os::WriteWholeFile(filename, j.dump(4));
}

Error PositionTrack::SaveCSV(std::string filename) {
	string csv;
	if (IsProjected)
		csv += tsf::fmt("%v,%v,%v\n", "x", "y", "time");
	else
		csv += tsf::fmt("%v,%v,%v\n", "lon", "lat", "time");

	for (const auto& v : Values)
		csv += tsf::fmt("%.6f,%.6f,%.3f\n", v.Pos.x, v.Pos.y, v.FrameTime);

	return os::WriteWholeFile(filename, csv);
}

bool PositionTrack::GetPositionAndVelocity(double frameTime, gfx::Vec3d& pos, gfx::Vec2d& vel2D) const {
	size_t i = algo::BinarySearchTry(Values.size(), &Values[0], frameTime, FindTime);
	i--;
	if (i == -1) {
		i         = 0;
		frameTime = 0;
	} else if (i > Values.size() - 2) {
		i         = Values.size() - 2;
		frameTime = Values[Values.size() - 1].FrameTime;
	}

	const auto& p1   = Values[i];
	const auto& p2   = Values[i + 1];
	double      frac = (frameTime - p1.FrameTime) / (p2.FrameTime - p1.FrameTime);

	Vec3d pp1, pp2;
	if (IsProjected) {
		pp1 = p1.Pos;
		pp2 = p2.Pos;
	} else {
		auto pp1 = OutProj.Convert(p1.Pos);
		auto pp2 = OutProj.Convert(p2.Pos);
	}

	pos            = pp1 + frac * (pp2 - pp1);
	auto   delta2D = (pp2 - pp1).vec2;
	double speed   = delta2D.size() / (p2.FrameTime - p1.FrameTime);
	vel2D          = speed * delta2D.normalized();

	return true;
}

gfx::Vec3d PositionTrack::GetPosition(double time) const {
	Vec3d pos;
	Vec2d vel2D;
	GetPositionAndVelocity(time, pos, vel2D);
	return pos;
}

void PositionTrack::Dump(double start, double end, double interval) const {
	for (double t = start; t < end; t += interval) {
		Vec3d pos;
		Vec2d vel;
		GetPositionAndVelocity(t, pos, vel);
		tsf::print("%4.1f, %12.1f %12.1f, %6.3f %6.3f (%6.3f)\n", t, pos.x, pos.y, vel.x, vel.y, vel.size());
	}
}

void PositionTrack::ConvertToWebMercator() {
	if (IsProjected)
		return;
	for (auto& v : Values)
		v.Pos = OutProj.Convert(v.Pos);
	IsProjected = true;
}

void PositionTrack::Simplify(double tolerance) {
	vector<Vec2d> pos;
	pos.resize(Values.size());
	for (size_t i = 0; i < Values.size(); i++)
		pos[i] = Values[i].Pos.vec2;

	bool* keep = new bool[pos.size()];
	gfx::Simplify(tolerance, pos.size(), &pos[0], keep);
	vector<TimePos> newVal;
	for (size_t i = 0; i < pos.size(); i++) {
		if (keep[i])
			newVal.push_back(Values[i]);
	}
	delete[] keep;
	Values = std::move(newVal);
}

void PositionTrack::Smooth(double smoothness, double increment) {
	vector<TimePos> newVal;
	newVal.push_back(Values[0]);

	Vec2d c1 = Values[0].Pos.vec2;
	Vec2d c2 = Values[1].Pos.vec2;

	size_t nM1 = Values.size() - 1;
	for (size_t i = 0; i < Values.size() - 1; i++) {
		Vec2d  c3, c4;
		Vec2d  p0      = Values[i].Pos.vec2;
		Vec2d  p1      = Values[i + 1].Pos.vec2;
		Vec2d  p2      = Values[min(i + 2, nM1)].Pos.vec2;
		double segDist = p1.distance2D(p0);
		if (segDist < increment) {
			newVal.push_back(Values[i + 1]);
		} else {
			ComputeSmoothCubicBezierControlPoints(p0.x, p0.y,
			                                      p1.x, p1.y,
			                                      p2.x, p2.y,
			                                      smoothness,
			                                      c3.x, c3.y,
			                                      c4.x, c4.y);

			int    pieces = (int) max(2.0, segDist / increment);
			double step   = 1.0 / pieces;
			double t      = step;
			size_t start  = newVal.size() - 1;
			for (int j = 0; j < pieces - 1; j++) {
				TimePos val  = Values[i];
				val.Pos.vec2 = EvaluateCubicBezier(p0, c2, c3, p1, t);
				newVal.push_back(val);
				t += step;
			}
			newVal.push_back(Values[i + 1]);
			// measure new length of curve, and interpolate time across it, so that we have constant velocity
			double totalLength = 0;
			for (int j = 0; j < pieces; j++) {
				totalLength += newVal[start + j].Pos.vec2.distance(newVal[start + j + 1].Pos.vec2);
			}
			t = 0;
			for (int j = 0; j < pieces; j++) {
				t += newVal[start + j].Pos.vec2.distance(newVal[start + j + 1].Pos.vec2) / totalLength;
				newVal[start + j].FrameTime = Values[i].FrameTime + t * (Values[i + 1].FrameTime - Values[i].FrameTime);
				newVal[start + j].Pos.z     = Values[i].Pos.z + t * (Values[i + 1].Pos.z - Values[i].Pos.z);
			}
		}
		c1 = c3;
		c2 = c4;
	}
	Values = std::move(newVal);
}

} // namespace roadproc
} // namespace imqs