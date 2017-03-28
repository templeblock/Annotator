#pragma once

namespace imqs {
namespace anno {

struct VideoStreamInfo {
	int Width;
	int Height;
};

class VideoFile {
public:
	static void Initialize();

	static StaticError ErrNeedMoreData; // Codec needs more data before it can deliver a frame/audio

	VideoFile();
	~VideoFile();

	void            Close();
	Error           OpenFile(std::string filename);
	VideoStreamInfo GetVideoStreamInfo();
	Error           DecodeFrameRGBA(int width, int height, void* buf, int stride);

private:
	AVFormatContext* FmtCtx         = nullptr;
	AVCodecContext*  VideoDecCtx    = nullptr;
	AVStream*        VideoStream    = nullptr;
	int              VideoStreamIdx = -1;
	AVFrame*         Frame          = nullptr;
	SwsContext*      SwsCtx         = nullptr;
	int              SwsDstW        = 0;
	int              SwsDstH        = 0;

	static Error TranslateErr(int ret, const char* whileBusyWith = nullptr);
	static Error OpenCodecContext(AVFormatContext* fmt_ctx, AVMediaType type, int& stream_idx, AVCodecContext*& dec_ctx);
	Error        RecvFrame();
	void         FlushCachedFrames();
};
}
}