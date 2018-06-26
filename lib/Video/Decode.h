#pragma once

namespace imqs {
namespace video {

struct IMQS_VIDEO_API VideoStreamInfo {
	int64_t    Duration  = 0; // AVFormatContext.duration
	int64_t    NumFrames = 0; // AvStream.nb_frames
	AVRational FrameRate;     // AVStream.r_frame_rate
	int        Width  = 0;
	int        Height = 0;

	double DurationSeconds() const;
	double FrameRateSeconds() const; // eg 29.97, or 24, etc
};

class IMQS_VIDEO_API VideoFile {
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
	Error           SeekToPreviousFrame();
	Error           SeekToFrame(int64_t frame);
	Error           SeekToFraction(double fraction_0_to_1);
	Error           SeekToSecond(double second);
	Error           SeekToMicrosecond(int64_t microsecond);
	double          LastFrameTimeSeconds() const;
	int64_t         LastFrameTimeMicrosecond() const;
	Error           DecodeFrameRGBA(int width, int height, void* buf, int stride);

	void Dimensions(int& width, int& height) const;

	int Width() const { return VideoDecCtx->width; }
	int Height() const { return VideoDecCtx->height; }

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
	int64_t          LastFramePTS   = 0;  // PTS = presentation time stamp (ie time when frame should be shown to user)
	int64_t          LastSeekPTS    = -1; // Set to a value other than -1, if we have just executed a seek operation

	static Error TranslateErr(int ret, const char* whileBusyWith = nullptr);
	static Error OpenCodecContext(AVFormatContext* fmt_ctx, AVMediaType type, int& stream_idx, AVCodecContext*& dec_ctx);
	Error        RecvFrame();
	void         FlushCachedFrames();
};

inline void VideoFile::Dimensions(int& width, int& height) const {
	if (VideoDecCtx) {
		width  = VideoDecCtx->width;
		height = VideoDecCtx->height;
	} else {
		width  = 0;
		height = 0;
	}
}

} // namespace video
} // namespace imqs