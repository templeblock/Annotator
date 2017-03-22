#pragma once

namespace imqs {
namespace anno {

class VideoFile {
public:
	static void Initialize();

	VideoFile();
	~VideoFile();

	void  Close();
	Error OpenFile(std::string filename);
	Error DecodeNextFrame(AVFrame*& frame);

private:
	AVFormatContext* FmtCtx         = nullptr;
	AVCodecContext*  VideoDecCtx    = nullptr;
	AVStream*        VideoStream    = nullptr;
	int              VideoStreamIdx = -1;
	AVFrame*         Frame          = nullptr;

	static Error OpenCodecContext(int& stream_idx, AVFormatContext* fmt_ctx, AVMediaType type);
	Error DecodePacket(AVPacket& pkt, bool& got_frame, int& decoded);
};

void TestDecode(xo::DomCanvas* canvas);
}
}