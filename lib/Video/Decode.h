#pragma once

#include "IVideo.h"

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

struct IMQS_VIDEO_API VideoMetadata {
	time::Time CreationTime;
};

class IMQS_VIDEO_API VideoFile : public IVideo {
public:
	static void Initialize();

	VideoFile();
	~VideoFile() override;

	void            Close();
	bool            IsOpen() const { return Filename != ""; }
	std::string     GetFilename() const { return Filename; }
	VideoStreamInfo GetVideoStreamInfo();
	VideoMetadata   Metadata();
	Error           SeekToPreviousFrame();
	Error           SeekToFrame(int64_t frame, unsigned flags = Seek::None);
	Error           SeekToFraction(double fraction_0_to_1, unsigned flags = Seek::None);
	Error           SeekToSecond(double second, unsigned flags = Seek::None);
	double          LastFrameTimeSeconds() const;
	int64_t         LastFrameTimeMicrosecond() const;

	// IVideo
	Error OpenFile(std::string filename) override;
	void  Info(int& width, int& height) override;
	Error DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds = nullptr) override;
	Error SeekToMicrosecond(int64_t microsecond, unsigned flags = Seek::None) override;

	void Dimensions(int& width, int& height) const;

	int Width() const { return VideoDecCtx->width; }
	int Height() const { return VideoDecCtx->height; }

private:
	std::string      Filename;
	AVFormatContext* FmtCtx         = nullptr;
	AVCodecContext*  VideoDecCtx    = nullptr;
	AVStream*        VideoStream    = nullptr;
	int              VideoStreamIdx = -1;
	AVPacket         Pkt;
	AVFrame*         Frame        = nullptr;
	SwsContext*      SwsCtx       = nullptr;
	int              SwsDstW      = 0;
	int              SwsDstH      = 0;
	int64_t          LastFramePTS = 0;  // PTS = presentation time stamp (ie time when frame should be shown to user)
	int64_t          LastSeekPTS  = -1; // Set to a value other than -1, if we have just executed a seek operation

	static Error TranslateErr(int ret, const char* whileBusyWith = nullptr);
	static Error OpenCodecContext(AVFormatContext* fmt_ctx, AVMediaType type, int& stream_idx, AVCodecContext*& dec_ctx);
	Error        RecvFrame();
	void         FlushCachedFrames();
	double       PtsToSeconds(int64_t pts) const;
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