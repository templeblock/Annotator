#include "pch.h"
#include "Globals.h"
#include "Perspective.h"

namespace imqs {
namespace roadproc {
int Speed(argparse::Args& args);
int Stitch(argparse::Args& args);
} // namespace roadproc
} // namespace imqs

// TODO: move into maps/tests/gfx/raster.cpp
namespace imqs {
namespace gfx {
namespace raster {
void TestBilinear() {
	{
		// 16-bit two channel (used for image warping)
		uint16_t img[8] = {
		    0x0001, 0x0002, 0x0050, 0x0060, //
		    0x0888, 0x0999, 0xffff, 0xfff0, //
		};
		auto show = [&](int32_t u, int32_t v) {
			uint32_t w = imqs::gfx::raster::ImageBilinear_RG_U16(img, 2, 255, 255, u, v);
			uint16_t x = w & 0xffff;
			uint16_t y = w >> 16;
			tsf::print("%v %v -> %02x %02x\n", u, v, x, y);
		};
		show(0, 0);
		show(255, 255);
		show(127, 127);
		show(0, 64);
	}

	{
		// 24-bit two channel (used for image warping)
		uint32_t img[8] = {
		    0x000100, 0x000200, 0x005000, 0x006000, //
		    0x088800, 0x099900, 0xffffff, 0xffffff, //
		};
		auto show = [&](int32_t u, int32_t v) {
			uint64_t w = imqs::gfx::raster::ImageBilinear_RG_U24(img, 2, 255, 255, u, v);
			uint32_t x = w & 0xffffffff;
			uint32_t y = w >> 32;
			tsf::print("%v %v -> %02x %02x\n", u, v, x, y);
		};
		show(0, 0);
		show(255, 255);
		show(127, 127);
		show(0, 64);
	}
}
} // namespace raster
} // namespace gfx
} // namespace imqs

int main(int argc, char** argv) {
	using namespace imqs::roadproc;

	imqs::video::VideoFile::Initialize();
	//imqs::gfx::raster::TestBilinear();

	argparse::Args args("Usage: RoadProcessor [options] <command>");
	args.AddValue("e", "lensdb", "Camera/Lens database", "/usr/local/share/lensfun/version_2/");
	args.AddValue("l", "lens", "Lens correction (eg 'Fujifilm X-T2,Samyang 12mm f/2.0 NCS CS'");

	auto speed = args.AddCommand("speed <video[,video2][...]>",
	                             "Compute car speed from interframe differences\nOne or more videos can be specified."
	                             " Separate multiple videos with commas.",
	                             Speed);
	speed->AddSwitch("", "csv", "Write CSV output (otherwise JSON)");
	speed->AddValue("o", "outfile", "Write output to file", "stdout");

	auto perspective = args.AddCommand("perspective <video>", "Compute perspective projection parameter z2.", Perspective);
	auto stitch      = args.AddCommand("stitch <video> <z2>", "Unproject video frames and stitch together.", Stitch);
	stitch->AddValue("n", "number", "Number of frames", "2");
	stitch->AddValue("s", "start", "Start time in seconds", "0");

	if (!args.Parse(argc, (const char**) argv))
		return 1;

	if (args.Get("lens") != "") {
		global::Lens = new LensCorrector();
		auto err     = global::Lens->LoadDatabase(args.Get("lensdb"));
		if (!err.OK()) {
			tsf::print("Error loading lens database: %v\n", err.Message());
			return 1;
		}
		err = global::Lens->LoadCameraAndLens(args.Get("lens"));
		if (!err.OK()) {
			tsf::print("Error loading camera/lens: %v\n", err.Message());
			return 1;
		}
	}

	int ret = args.ExecCommand();

	delete global::Lens;

	return ret;
}
