#pragma once

namespace imqs {
namespace anno {

struct VideoStreamInfo {
	int64_t    Duration  = 0; // AVFormatContext.duration
	int64_t    NumFrames = 0; // AvStream.nb_frames
	AVRational FrameRate;     // AVStream.r_frame_rate
	int        Width  = 0;
	int        Height = 0;

	double DurationSeconds() const;
	double FrameRateSeconds() const;
};

class VideoFile {
public:
	static void Initialize();

	static StaticError ErrNeedMoreData; // Codec needs more data before it can deliver a frame/audio

	VideoFile();
	~VideoFile();

	void            Close();
	bool            IsOpen() const { return Filename != ""; }
	Error           OpenFile(std::string filename);
	std::string     GetFilename() const { return Filename; }
	VideoStreamInfo GetVideoStreamInfo();
	Error           SeekToFrame(int64_t frame);
	Error           SeekToFraction(double fraction_0_to_1);
	double          LastFrameTimeSeconds();
	Error           DecodeFrameRGBA(int width, int height, void* buf, int stride);

private:
	std::string      Filename;
	AVFormatContext* FmtCtx         = nullptr;
	AVCodecContext*  VideoDecCtx    = nullptr;
	AVStream*        VideoStream    = nullptr;
	int              VideoStreamIdx = -1;
	AVFrame*         Frame          = nullptr;
	SwsContext*      SwsCtx         = nullptr;
	int              SwsDstW        = 0;
	int              SwsDstH        = 0;
	int64_t          LastFramePTS   = 0; // PTS = presentation time stamp (ie time when frame should be shown to user)

	static Error TranslateErr(int ret, const char* whileBusyWith = nullptr);
	static Error OpenCodecContext(AVFormatContext* fmt_ctx, AVMediaType type, int& stream_idx, AVCodecContext*& dec_ctx);
	Error        RecvFrame();
	void         FlushCachedFrames();
};
}
}