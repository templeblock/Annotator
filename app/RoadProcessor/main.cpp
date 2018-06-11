#include "pch.h"

namespace imqs {
namespace roadproc {
int Speed(argparse::Args& args);
}
} // namespace imqs

int main(int argc, char** argv) {
	using namespace imqs::roadproc;

	imqs::video::VideoFile::Initialize();

	argparse::Args args("Usage: RoadProcessor [options] <command>");
	auto           speed = args.AddCommand("speed <video>", "Compute car speed from interframe differences", Speed);
	speed->AddSwitch("", "csv", "Write CSV output");
	speed->AddValue("o", "outfile", "Write output to file", "stdout");
	if (!args.Parse(argc, (const char**) argv))
		return 1;
	return args.ExecCommand();
}
