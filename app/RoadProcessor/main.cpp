#include "pch.h"

namespace imqs {
namespace roadproc {
int Speed(argparse::Args& args);
int Perspective(argparse::Args& args);
} // namespace roadproc
} // namespace imqs

int main(int argc, char** argv) {
	using namespace imqs::roadproc;

	imqs::video::VideoFile::Initialize();

	argparse::Args args("Usage: RoadProcessor [options] <command>");
	auto           speed = args.AddCommand("speed <video[,video2][...]>",
                                 "Compute car speed from interframe differences\nOne or more videos can be specified."
                                 " Separate multiple videos with commas.",
                                 Speed);
	speed->AddSwitch("", "csv", "Write CSV output (otherwise JSON)");
	speed->AddValue("o", "outfile", "Write output to file", "stdout");

	auto perspective = args.AddCommand("perspective <video>",
	                                   "Compute perspective projection.",
	                                   Perspective);

	if (!args.Parse(argc, (const char**) argv))
		return 1;
	return args.ExecCommand();
}
