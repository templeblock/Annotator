#include "pch.h"
#include "Decode.h"

namespace imqs {
namespace video {

double VideoStreamInfo::DurationSeconds() const {
	return (double) Duration / (double) AV_TIME_BASE;
}

double VideoStreamInfo::FrameRateSeconds() const {
	return av_q2d(FrameRate);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void VideoFile::Initialize() {
	av_register_all();
}

VideoFile::VideoFile() {
	memset(&Pkt, 0, sizeof(Pkt));
}

VideoFile::~VideoFile() {
	Close();
}

void VideoFile::Close() {
	FlushCachedFrames();

	av_packet_unref(&Pkt);
	sws_freeContext(SwsCtx);
	avcodec_free_context(&VideoDecCtx);
	avformat_close_input(&FmtCtx);
	av_frame_free(&Frame);
	FmtCtx         = nullptr;
	VideoDecCtx    = nullptr;
	VideoStream    = nullptr;
	VideoStreamIdx = -1;
	Frame          = nullptr;
	SwsCtx         = nullptr;
	SwsDstW        = 0;
	SwsDstH        = 0;
	Filename       = "";
}

Error VideoFile::OpenFile(std::string filename) {
	Close();

	int r = avformat_open_input(&FmtCtx, filename.c_str(), NULL, NULL);
	if (r < 0)
		return TranslateAvErr(r, tsf::fmt("Could not open video file %v", filename).c_str());

	r = avformat_find_stream_info(FmtCtx, NULL);
	if (r < 0) {
		Close();
		return TranslateAvErr(r, "Could not find stream information");
	}

	auto err = OpenCodecContext(FmtCtx, AVMEDIA_TYPE_VIDEO, VideoStreamIdx, VideoDecCtx);
	if (!err.OK()) {
		Close();
		return err;
	}

	VideoStream = FmtCtx->streams[VideoStreamIdx];

	//av_dump_format(fmt_ctx, 0, src_filename, 0);

	Frame = av_frame_alloc();
	if (!Frame) {
		Close();
		return Error("Out of memory allocating frame");
	}

	av_init_packet(&Pkt);
	Pkt.data = nullptr;
	Pkt.size = 0;

	Filename = filename;
	return Error();
}

void VideoFile::Info(int& width, int& height) {
	width  = Width();
	height = Height();
}

VideoStreamInfo VideoFile::GetVideoStreamInfo() {
	VideoStreamInfo inf;
	// frame rate: 119.880116
	// duration: 4:31
	//int64_t duration = FmtCtx->duration;
	//int64_t tbase = VideoDecCtx->
	//av_rescale()
	inf.Duration  = FmtCtx->duration;
	inf.NumFrames = VideoStream->nb_frames;
	inf.FrameRate = VideoStream->r_frame_rate;
	Dimensions(inf.Width, inf.Height);
	return inf;
}

VideoMetadata VideoFile::Metadata() {
	VideoMetadata      md;
	AVDictionaryEntry* v = av_dict_get(VideoStream->metadata, "creation_time", nullptr, 0);
	if (v)
		md.CreationTime.Parse8601(v->value, strlen(v->value));
	return md;
}

Error VideoFile::SeekToPreviousFrame() {
	auto timePerFrame = av_inv_q(VideoStream->avg_frame_rate);
	auto t            = av_mul_q({(int) LastFramePTS, 1}, VideoStream->time_base);
	t                 = av_sub_q(t, timePerFrame);

	if (t.num < 0)
		t.num = 0;

	t = av_div_q(t, VideoStream->time_base);
	// I expect den to always be 1 here, but not sure
	IMQS_ASSERT(t.den == 1);
	int64_t pts = t.num / t.den;

	int r = av_seek_frame(FmtCtx, VideoStreamIdx, pts, AVSEEK_FLAG_BACKWARD);
	if (r >= 0)
		LastSeekPTS = pts;
	return Error();
}

Error VideoFile::SeekToFrame(int64_t frame, unsigned flags) {
	// frame rate = number of frames per second. So to get seconds/frame, we divide by frame rate.
	auto t = av_div_q({(int) frame, 1}, VideoStream->avg_frame_rate);

	// divide by time_base to get PTS
	auto tb = av_div_q(t, VideoStream->time_base);

	// I'm not sure that av_div_q will always give denominator of 1, but the fraction
	// *should* be exact.
	int64_t pts = tb.num / tb.den;

	int r = av_seek_frame(FmtCtx, VideoStreamIdx, pts, flags);
	if (r >= 0)
		LastSeekPTS = pts;
	return Error();
}

Error VideoFile::SeekToFraction(double fraction_0_to_1, unsigned flags) {
	double seconds = fraction_0_to_1 * GetVideoStreamInfo().DurationSeconds();
	return SeekToMicrosecond((int64_t)(seconds * 1000000.0), flags);
}

Error VideoFile::SeekToSecond(double second, unsigned flags) {
	return SeekToMicrosecond((int64_t)(second * 1000000.0), flags);
}

Error VideoFile::SeekToMicrosecond(int64_t microsecond, unsigned flags) {
	double  ts  = av_q2d(VideoStream->time_base);
	double  t   = ((double) microsecond / 1000000.0) / ts;
	int64_t pts = (int64_t) t;
	int     r   = av_seek_frame(FmtCtx, VideoStreamIdx, pts, flags);
	if (r >= 0 && !!(flags & Seek::Any))
		LastSeekPTS = pts;
	return Error();
}

double VideoFile::LastFrameTimeSeconds() const {
	return PtsToSeconds(LastFramePTS);
}

int64_t VideoFile::LastFrameTimeMicrosecond() const {
	return (int64_t)(LastFrameTimeSeconds() * 1000000.0);
}

double VideoFile::PtsToSeconds(int64_t pts) const {
	return av_q2d(av_mul_q({(int) pts, 1}, VideoStream->time_base));
}

/* This turns out to be useless, because the FFMPeg rationals are stored as
32-bit num/dem, so with a denominator of 1000000 you quickly hit 32-bit limits.
int64_t VideoFile::LastFrameAVTime() const {
	// change the time unit to AV_TIME_BASE
	auto v1 = av_make_q(VideoStream->time_base.num, VideoStream->time_base.den);
	auto v2 = av_make_q(AV_TIME_BASE, 1);
	auto scale = av_mul_q(v1, v2);
	auto a = av_mul_q({(int) LastFramePTS, 1}, scale);
	return a.num;
}
*/

Error VideoFile::DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds) {
	// Allow for multiple attempts, if we have just performed a seek.
	// After performing a seek, the codec will often emit what looks like a previously buffered
	// frame. If the first frame that we receive is not the one that we seeked to, then
	// we throw that frame away.
	// NOTE: This can be very expensive, if you are far ahead of your last keyframe.
	// However, sometimes that's what the user wants.
	int wasBehind = 0;

	for (int attempt = 0; attempt < 200; attempt++) {
		bool haveFrame = false;
		int  nReceive  = 0;
		while (!haveFrame) {
			int r = avcodec_receive_frame(VideoDecCtx, Frame);
			nReceive++;
			switch (r) {
			case 0:
				haveFrame = true;
				break;
			case AVERROR_EOF:
				return ErrEOF;
			case AVERROR(EAGAIN): {
				// need more data
				AVPacket pkt;
				av_init_packet(&pkt);
				pkt.data = nullptr;
				pkt.size = 0;
				r        = av_read_frame(FmtCtx, &pkt);
				if (r != 0)
					return TranslateAvErr(r, "av_read_frame");
				r = avcodec_send_packet(VideoDecCtx, &pkt);
				av_packet_unref(&pkt);
				if (r == AVERROR_INVALIDDATA) {
					// skip over invalid data, and keep trying
				} else if (r != 0) {
					return TranslateAvErr(r, "avcodec_send_packet");
				}
				break;
			}
			default:
				return TranslateAvErr(r, "avcodec_receive_frame");
			}
		}
		IMQS_ASSERT(haveFrame);

		double secondsDelta = PtsToSeconds(Frame->pts - LastFramePTS);

		// This is supposed to determine whether this is old data that the codec is flushing, before
		// it performs a seek that we requested.
		bool isLikelyOldBufferedFrame = attempt == 0 && LastSeekPTS != -1 && secondsDelta > 0 && secondsDelta < 1.0;

		LastFramePTS = Frame->pts;

		//auto msg     = tsf::fmt("LastFramePTS: %d\n", (int) LastFramePTS);
		//OutputDebugStringA(msg.c_str());

		// If there was no recent seek operation, then just accept whatever we get
		if (LastSeekPTS == -1)
			break;

		// If there was a seek operation, then allow multiple attempts until we get to the frame we want
		if (LastFramePTS == LastSeekPTS)
			break;

		// If the seek operation ended up putting us AHEAD of our desired seek position, then accept it
		if (LastFramePTS > LastSeekPTS && !isLikelyOldBufferedFrame)
			break;

		bool isBehind = LastFramePTS < LastSeekPTS;
		if (wasBehind && !isBehind) {
			// We started decoding behind our seek position, and now we've decoded ahead of it, so this means that
			// our seek position was not exact. This is the appropriate place to jump out (or perhaps, one frame back,
			// but this is probably fine).
			break;
		}
		wasBehind = isBehind;
	}

	LastSeekPTS = -1;
	if (timeSeconds)
		*timeSeconds = LastFrameTimeSeconds();

	if (width == 0 || height == 0 || buf == nullptr) {
		// caller is not interested in actual frame pixels
		return Error();
	}

	if (SwsCtx && ((SwsDstW != width) || (SwsDstH != height))) {
		sws_freeContext(SwsCtx);
		SwsCtx = nullptr;
	}

	if (!SwsCtx) {
		SwsCtx = sws_getContext(Frame->width, Frame->height, (AVPixelFormat) Frame->format, width, height, AVPixelFormat::AV_PIX_FMT_RGBA, 0, nullptr, nullptr, nullptr);
		if (!SwsCtx)
			return Error("Unable to create libswscale scaling context");
		SwsDstH = height;
		SwsDstW = width;
	}

	uint8_t* buf8         = (uint8_t*) buf;
	uint8_t* dst[4]       = {buf8 + 0, buf8 + 1, buf8 + 2, buf8 + 3};
	int      dstStride[4] = {stride, stride, stride, stride};

	sws_scale(SwsCtx, Frame->data, Frame->linesize, 0, Frame->height, dst, dstStride);

	return Error();
}

Error VideoFile::RecvFrame() {
	int ret = avcodec_receive_frame(VideoDecCtx, Frame);
	if (ret == 0)
		return Error();
	return TranslateAvErr(ret, "avcodec_receive_frame");
}

Error VideoFile::OpenCodecContext(AVFormatContext* fmt_ctx, AVMediaType type, int& stream_idx, AVCodecContext*& dec_ctx) {
	int           ret;
	AVStream*     st;
	AVCodec*      dec  = NULL;
	AVDictionary* opts = NULL;

	ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
	if (ret < 0)
		return Error::Fmt("Could not find %s stream", av_get_media_type_string(type));

	stream_idx = ret;
	st         = fmt_ctx->streams[stream_idx];

	// find decoder for the stream
	dec = avcodec_find_decoder(st->codecpar->codec_id);
	if (!dec)
		return Error::Fmt("Failed to find %s codec", av_get_media_type_string(type));

	// Dies during avcodec_send_packet with "av_bsf_receive_packet failed"
	// I cannot find any information about this on the internet.

	//if (strcmp(dec->name, "h264") == 0) {
	//	AVCodec* decHW = avcodec_find_decoder_by_name("h264_cuvid");
	//	if (decHW)
	//		dec = decHW;
	//}
	//tsf::print("Using decoder: %v (%v)\n", dec->name, dec->long_name);

	// Allocate a codec context for the decoder
	dec_ctx = avcodec_alloc_context3(dec);
	if (!dec_ctx)
		return Error::Fmt("Failed to allocate %v codec context", av_get_media_type_string(type));

	// Copy codec parameters from input stream to output codec context
	ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
	if (ret < 0)
		return Error::Fmt("Failed to copy %s codec parameters to decoder context. Error %v", av_get_media_type_string(type), ret);

	// Init the video decoder
	//av_dict_set(&opts, "refcounted_frames", "1", 0); // not necessary, since avcodec_receive_frame() always uses refcounted frames
	//av_dict_set(&opts, "flags2", "+export_mvs", 0); // motion vectors
	ret = avcodec_open2(dec_ctx, dec, &opts);
	if (ret < 0)
		return Error::Fmt("Failed to open %s codec", av_get_media_type_string(type));

	return Error();
}

void VideoFile::FlushCachedFrames() {
	// flush cached frames in the codec's buffers
	if (!VideoDecCtx)
		return;

	// send an empty packet which instructs the codec to start flushing
	AVPacket pkt;
	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;
	avcodec_send_packet(VideoDecCtx, &pkt);

	// drain the codec
	while (true) {
		int r = avcodec_receive_frame(VideoDecCtx, Frame);
		if (r != 0)
			break;
	}
}
} // namespace video
} // namespace imqs