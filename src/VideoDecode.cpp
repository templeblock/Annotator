#include "pch.h"
#include "VideoDecode.h"

namespace imqs {
namespace anno {

void VideoFile::Initialize() {
	av_register_all();
}

VideoFile::VideoFile() {
}

VideoFile::~VideoFile() {
	Close();
}

void VideoFile::Close() {
	// TODO? flush cached frames
	// pkt.data = NULL;
	// pkt.size = 0;
	// do {
	// 	decode_packet(&got_frame, 1);
	// } while (got_frame);

	avcodec_close(VideoDecCtx);
	avformat_close_input(&FmtCtx);
	av_frame_free(&Frame);
}

Error VideoFile::OpenFile(std::string filename) {
	Close();

	//src_filename = "D:\\mldata\\GOPR0080.MP4";

	if (avformat_open_input(&FmtCtx, filename.c_str(), NULL, NULL) < 0)
		return Error::Fmt("Could not open video file %v", filename);

	if (avformat_find_stream_info(FmtCtx, NULL) < 0) {
		Close();
		return Error("Could not find stream information");
	}

	auto err = OpenCodecContext(VideoStreamIdx, FmtCtx, AVMEDIA_TYPE_VIDEO);
	if (!err.OK()) {
		Close();
		return err;
	}

	VideoStream = FmtCtx->streams[VideoStreamIdx];
	VideoDecCtx = VideoStream->codec;

	//av_dump_format(fmt_ctx, 0, src_filename, 0);

	if (!VideoStream) {
		Close();
		return Error("Could not find video stream in the input");
	}

	Frame = av_frame_alloc();
	if (!Frame) {
		Close();
		return Error("Out of memory allocating frame");
	}

	return Error();
}

Error VideoFile::DecodeNextFrame(AVFrame*& frame) {
	// initialize packet, set data to NULL, let the demuxer fill it
	AVPacket pkt;
	av_init_packet(&pkt);
	pkt.data = nullptr;
	pkt.size = 0;
	
	bool got_frame = false;

	while (av_read_frame(FmtCtx, &pkt) == 0) {
		AVPacket orig_pkt = pkt;
		do {
			int  decoded   = 0;
			auto err       = DecodePacket(pkt, got_frame, decoded);
			// this looks like a leak, because we skip av_packet_unref
			if (!err.OK())
				return err;
			pkt.data += decoded;
			pkt.size -= decoded;
		} while (pkt.size > 0 && !got_frame);
		av_packet_unref(&orig_pkt);
		if (got_frame)
			break;
	}
	if (!got_frame)
		return Error("No more frames");

	frame = Frame;
	return Error();
}

Error VideoFile::OpenCodecContext(int& stream_idx, AVFormatContext* fmt_ctx, AVMediaType type) {
	int             ret;
	AVStream*       st;
	AVCodecContext* dec_ctx = NULL;
	AVCodec*        dec     = NULL;
	AVDictionary*   opts    = NULL;

	ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
	if (ret < 0) {
		return Error::Fmt("Could not find %s stream", av_get_media_type_string(type));
	} else {
		stream_idx = ret;
		st         = fmt_ctx->streams[stream_idx];

		/* find decoder for the stream */
		dec_ctx = st->codec;
		dec     = avcodec_find_decoder(dec_ctx->codec_id);
		if (!dec) {
			return Error::Fmt("Failed to find %s codec", av_get_media_type_string(type));
		}

		/* Init the video decoder */
		av_dict_set(&opts, "flags2", "+export_mvs", 0);
		ret = avcodec_open2(dec_ctx, dec, &opts);
		if (ret < 0) {
			return Error::Fmt("Failed to open %s codec", av_get_media_type_string(type));
		}
	}

	return Error();
}

Error VideoFile::DecodePacket(AVPacket& pkt, bool& got_frame, int& decoded) {
	decoded   = pkt.size;
	got_frame = false;

	if (pkt.stream_index == VideoStreamIdx) {
		int igot_frame = 0;
		int ret        = avcodec_decode_video2(VideoDecCtx, Frame, &igot_frame, &pkt);
		if (ret < 0)
			return Error::Fmt("Error decoding video frame (%v)", ret);
		got_frame = igot_frame != 0;

		/*
		if (got_frame) {
			static int video_frame_count = 0;
			video_frame_count++;
			AVFrameSideData* sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
			if (sd) {
				const AVMotionVector* mvs = (const AVMotionVector*) sd->data;
				for (int i = 0; i < sd->size / sizeof(*mvs); i++) {
					const AVMotionVector* mv = &mvs[i];
					printf("%d,%2d,%2d,%2d,%4d,%4d,%4d,%4d,0x%" PRIx64 "\n",
					       video_frame_count, mv->source,
					       mv->w, mv->h, mv->src_x, mv->src_y,
					       mv->dst_x, mv->dst_y, mv->flags);
				}
			}
		}
		*/
	}

	return Error();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static AVFormatContext* fmt_ctx       = NULL;
static AVCodecContext*  video_dec_ctx = NULL;
static AVStream*        video_stream  = NULL;
static const char*      src_filename  = NULL;

static int      video_stream_idx = -1;
static AVFrame* frame            = NULL;
static AVPacket pkt;
static int      video_frame_count = 0;

static int decode_packet(int* got_frame, int cached) {
	int decoded = pkt.size;

	*got_frame = 0;

	if (pkt.stream_index == video_stream_idx) {
		int ret = avcodec_decode_video2(video_dec_ctx, frame, got_frame, &pkt);
		if (ret < 0) {
			//fprintf(stderr, "Error decoding video frame (%s)\n", av_err2str(ret));
			//auto errMsg = av_err2str(ret);
			fprintf(stderr, "Error decoding video frame (%d)\n", ret);
			return ret;
		}

		if (*got_frame) {
			int              i;
			AVFrameSideData* sd;

			video_frame_count++;
			sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
			if (sd) {
				const AVMotionVector* mvs = (const AVMotionVector*) sd->data;
				for (i = 0; i < sd->size / sizeof(*mvs); i++) {
					const AVMotionVector* mv = &mvs[i];
					printf("%d,%2d,%2d,%2d,%4d,%4d,%4d,%4d,0x%" PRIx64 "\n",
					       video_frame_count, mv->source,
					       mv->w, mv->h, mv->src_x, mv->src_y,
					       mv->dst_x, mv->dst_y, mv->flags);
				}
			}
		}
	}

	return decoded;
}

static int open_codec_context(int*             stream_idx,
                              AVFormatContext* fmt_ctx, enum AVMediaType type) {
	int             ret;
	AVStream*       st;
	AVCodecContext* dec_ctx = NULL;
	AVCodec*        dec     = NULL;
	AVDictionary*   opts    = NULL;

	ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
	if (ret < 0) {
		fprintf(stderr, "Could not find %s stream in input file '%s'\n",
		        av_get_media_type_string(type), src_filename);
		return ret;
	} else {
		*stream_idx = ret;
		st          = fmt_ctx->streams[*stream_idx];

		/* find decoder for the stream */
		dec_ctx = st->codec;
		dec     = avcodec_find_decoder(dec_ctx->codec_id);
		if (!dec) {
			fprintf(stderr, "Failed to find %s codec\n",
			        av_get_media_type_string(type));
			return AVERROR(EINVAL);
		}

		/* Init the video decoder */
		av_dict_set(&opts, "flags2", "+export_mvs", 0);
		if ((ret = avcodec_open2(dec_ctx, dec, &opts)) < 0) {
			fprintf(stderr, "Failed to open %s codec\n",
			        av_get_media_type_string(type));
			return ret;
		}
	}

	return 0;
}

void TestDecode(xo::DomCanvas* canvas) {
	VideoFile v;
	auto err = v.OpenFile("D:\\mldata\\GOPR0080.MP4");
	if (!err.OK())
		return;

	AVFrame* frame;
	err = v.DecodeNextFrame(frame);
	if (err.OK()) {
		auto cx = canvas->GetCanvas2D();
		size_t w = std::min<size_t>(cx->Width(), frame->width);
		size_t h = std::min<size_t>(cx->Height(), frame->height);
		for (size_t y = 0; y < h; y++) {
			const uint8_t* src = frame->data[0];
			src += y * (size_t) frame->linesize[0];
			uint8_t* dst = (uint8_t*) cx->RowPtr(y);
			for (size_t x = 0; x < w; x++) {
				dst[0] = src[0];
				dst[1] = src[0];
				dst[2] = src[0];
				dst[3] = 255;
				src++;
				dst += 4;
			}
		}
		cx->Invalidate();
		canvas->ReleaseCanvas(cx);
	}
}

void TestDecode_OLD(xo::DomCanvas* canvas) {
	int ret = 0, got_frame;

	src_filename = "D:\\mldata\\GOPR0080.MP4";

	av_register_all();

	if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0) {
		fprintf(stderr, "Could not open source file %s\n", src_filename);
		exit(1);
	}

	if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
		fprintf(stderr, "Could not find stream information\n");
		exit(1);
	}

	if (open_codec_context(&video_stream_idx, fmt_ctx, AVMEDIA_TYPE_VIDEO) >= 0) {
		video_stream  = fmt_ctx->streams[video_stream_idx];
		video_dec_ctx = video_stream->codec;
	}

	av_dump_format(fmt_ctx, 0, src_filename, 0);

	if (!video_stream) {
		fprintf(stderr, "Could not find video stream in the input, aborting\n");
		ret = 1;
		goto end;
	}

	frame = av_frame_alloc();
	if (!frame) {
		fprintf(stderr, "Could not allocate frame\n");
		ret = AVERROR(ENOMEM);
		goto end;
	}

	printf("framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags\n");

	/* initialize packet, set data to NULL, let the demuxer fill it */
	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;

	/* read frames from the file */
	while (av_read_frame(fmt_ctx, &pkt) >= 0) {
		AVPacket orig_pkt = pkt;
		do {
			ret = decode_packet(&got_frame, 0);
			if (ret < 0)
				break;
			pkt.data += ret;
			pkt.size -= ret;
		} while (pkt.size > 0);
		av_packet_unref(&orig_pkt);
	}

	/* flush cached frames */
	pkt.data = NULL;
	pkt.size = 0;
	do {
		decode_packet(&got_frame, 1);
	} while (got_frame);

end:
	avcodec_close(video_dec_ctx);
	avformat_close_input(&fmt_ctx);
	av_frame_free(&frame);
}
}
}