#include "pch.h"
#include "VideoDecode.h"

namespace imqs {
namespace anno {

StaticError VideoFile::ErrNeedMoreData("Codec needs more data");

void VideoFile::Initialize() {
	av_register_all();
}

VideoFile::VideoFile() {
}

VideoFile::~VideoFile() {
	Close();
}

void VideoFile::Close() {
	FlushCachedFrames();

	sws_freeContext(SwsCtx);
	avcodec_free_context(&VideoDecCtx);
	avformat_close_input(&FmtCtx);
	av_frame_free(&Frame);
}

Error VideoFile::OpenFile(std::string filename) {
	Close();

	//src_filename = "D:\\mldata\\GOPR0080.MP4";

	int r = avformat_open_input(&FmtCtx, filename.c_str(), NULL, NULL);
	if (r < 0)
		return TranslateErr(r, tsf::fmt("Could not open video file %v", filename).c_str());

	r = avformat_find_stream_info(FmtCtx, NULL);
	if (r < 0) {
		Close();
		return TranslateErr(r, "Could not find stream information");
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

	return Error();
}

VideoStreamInfo VideoFile::GetVideoStreamInfo() {
	VideoStreamInfo inf;
	inf.Width  = VideoDecCtx->width;
	inf.Height = VideoDecCtx->height;
	return inf;
}

Error VideoFile::DecodeFrameRGBA(int width, int height, void* buf, int stride) {
	bool haveFrame = false;
	while (!haveFrame) {
		int r = avcodec_receive_frame(VideoDecCtx, Frame);
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
				return TranslateErr(r, "av_read_frame");
			r = avcodec_send_packet(VideoDecCtx, &pkt);
			av_packet_unref(&pkt);
			if (r == AVERROR_INVALIDDATA) {
				// skip over invalid data, and keep trying
			} else if (r != 0) {
				return TranslateErr(r, "avcodec_send_packet");
			}
			break;
		}
		default:
			return TranslateErr(r, "avcodec_receive_frame");
		}
	}

	if (SwsCtx && (SwsDstW != width) || (SwsDstH != height)) {
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
	return TranslateErr(ret, "avcodec_receive_frame");
}

Error VideoFile::TranslateErr(int ret, const char* whileBusyWith) {
	char errBuf[AV_ERROR_MAX_STRING_SIZE + 1];

	switch (ret) {
	case AVERROR_EOF: return ErrEOF;
	case AVERROR(EAGAIN): return ErrNeedMoreData;
	default:
		av_strerror(ret, errBuf, sizeof(errBuf));
		if (whileBusyWith)
			return Error::Fmt("%v: Error %", whileBusyWith, errBuf);
		else
			return Error::Fmt("AVERROR %v", errBuf);
	}
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
}
}