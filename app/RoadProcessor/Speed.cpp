#include "pch.h"
#include "Globals.h"
#include "Perspective.h"
#include "MeshRenderer.h"
#include "OpticalFlow.h"
#include "VideoStitcher.h"
#include "Speed.h"

// This is the third version of our vehicle speed computation system, which uses
// optical flow on flattened images, instead of using OpenCV feature tracking.
//
// Time to do 3 mthata videos (4.1GB, 4.1GB, 3.8GB): 40 minutes
//
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' speed --csv -0.00095 ~/mldata/DSCF3040.MOV 2>/dev/null
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' speed -o speeds.json -0.00095 ~/mldata/DSCF3040.MOV 2>/dev/null
//
// mthata
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS'speed --csv -0.000411 ~/mldata/mthata/DSCF0001-HG-3.MOV 2>/dev/null
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' speed -o mthata-speeds.json -0.000411 /home/ben/mldata/mthata/DSCF0001-HG-3.MOV,/home/ben/mldata/mthata/DSCF0001-HG-4.MOV,/home/ben/mldata/mthata/DSCF0001-HG-5.MOV 2>/dev/null
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' speed --csv -0.000411 /home/ben/mldata/mthata/DSCF0001-HG-3.MOV,/home/ben/mldata/mthata/DSCF0001-HG-4.MOV,/home/ben/mldata/mthata/DSCF0001-HG-5.MOV 2>/dev/null
// build/run-roadprocessor -r --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' speed --csv -0.000411 /home/ben/mldata/mthata/Day3-4.MOV 2>/dev/null
// build/run-roadprocessor --lens 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS' speed --csv -0.000411 /home/ben/mldata/mthata/Day3-4.MOV
// Day3-11: ZY: -0.000794

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

Error DoSpeed(vector<string> videoFiles, float zy, double startTime, SpeedOutputMode outputMode, string outputFile) {
	FILE* outf = stdout;
	if (outputFile != "stdout") {
		outf = fopen(outputFile.c_str(), "w");
		if (!outf)
			return Error::Fmt("Failed to open output file '%v'", outputFile);
	}

	if (outputMode == SpeedOutputMode::CSV)
		tsf::print(outf, "time,speed\n");

	VideoStitcher stitcher;
	stitcher.StartVideoAt = startTime;
	auto err              = stitcher.Start(videoFiles, zy);
	if (!err.OK())
		return err;

	for (int frame = 0; true; frame++) {
		err = stitcher.Next();
		if (err == ErrEOF)
			break;
		else if (!err.OK())
			return err;

		if (outputMode == SpeedOutputMode::CSV && frame >= 1) {
			// we will be one frame behind at the start, because we only estimate frame 0's speed based on frame 1's speed.
			int history = frame == 1 ? -1 : 0;
			for (int i = history; i <= 0; i++)
				tsf::print("%.3f,%.1f\n", stitcher.Velocities[frame + i].first, stitcher.Velocities[frame + i].second.size());
		}

		if (frame % 10 == 0 && outf != stdout)
			stitcher.PrintRemainingTime();
	}

	if (outputMode == SpeedOutputMode::JSON) {
		nlohmann::json j;
		j["time"] = stitcher.FirstVideoCreationTime.UnixNano() / 1000000;
		nlohmann::json jspeed;
		for (const auto& p : stitcher.Velocities) {
			nlohmann::json jp;
			jp.push_back(p.first);
			jp.push_back(p.second.size());
			jspeed.push_back(std::move(jp));
		}
		j["speeds"] = std::move(jspeed);
		auto js     = j.dump(4);
		fwrite(js.c_str(), js.size(), 1, outf);
	}

	if (outf != stdout)
		fclose(outf);

	return Error();
}

int Speed(argparse::Args& args) {
	auto zy         = atof(args.Params[0].c_str());
	auto videoFiles = strings::Split(args.Params[1], ',');
	auto startTime  = atof(args.Get("start").c_str());
	auto err        = DoSpeed(videoFiles, zy, startTime, args.Has("csv") ? SpeedOutputMode::CSV : SpeedOutputMode::JSON, args.Get("outfile"));
	if (!err.OK()) {
		tsf::print(stderr, "Error: %v\n", err.Message());
		tsf::print("Error measuring speed: %v\n", err.Message());
		return 1;
	}
	return 0;
}

} // namespace roadproc
} // namespace imqs
