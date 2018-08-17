#include "pch.h"
#include "Bench.h"

using namespace std;
using namespace imqs::gfx;

namespace imqs {
namespace roadproc {

int Bench(argparse::Args& args) {
	BenchVideoDecode();
	return 0;
}

void BenchVideoDecode() {
	Error  err;
	string filename = "/home/ben/mldata/mthata/Day3-4.MOV";
	int    n        = 300;

	NVVideo nv;
	err = nv.Initialize();
	err = nv.OpenFile(filename);
	tsf::print("Open: %v\n", err.Message());
	IMQS_ASSERT(err.OK());
	Image img;
	img.Alloc(ImageFormat::RGBA, nv.Width(), nv.Height());
	auto start = time::Now();
	for (int i = 0; i < n; i++) {
		err = nv.DecodeFrameRGBA(img.Width, img.Height, img.Data, img.Stride);
		//img.SaveFile("frame-nvdec.jpg");
		IMQS_ASSERT(err.OK());
	}
	tsf::print("FPS: %f\n", n / (time::Now() - start).Seconds());

	/*
	video::VideoFile v;
	err = v.OpenFile(filename);
	IMQS_ASSERT(err.OK());
	img.Alloc(ImageFormat::RGBA, v.Width(), v.Height());
	start = time::Now();
	for (int i = 0; i < n; i++) {
		//err = v.DecodeFrameRGBA(img.Width, img.Height, img.Data, img.Stride);
		err = v.DecodeFrameRGBA(0, 0, nullptr, 0);
		IMQS_ASSERT(err.OK());
	}
	tsf::print("FPS: %f\n", n / (time::Now() - start).Seconds());
	*/
}

} // namespace roadproc
} // namespace imqs