#pragma once

namespace imqs {
namespace anno {

class VideoFile {
public:
	static void Initialize();

	Error OpenFile(std::string filename);

private:
	AVFormatContext* FmtCtx      = nullptr;
	AVCodecContext*  VideoDecCtx = nullptr;
	AVStream*        VideoStream = nullptr;
};

void TestDecode(xo::DomCanvas* canvas);
}
}