#include "pch.h"
#include "FFmpegDemuxer.h"
#include "../../IVideo.h"

namespace imqs {
namespace video {

static bool AreCodecsRegistered = false;

FFmpegDemuxer::FFmpegDemuxer() {
	memset(&pkt, 0, sizeof(pkt));
	memset(&pktFiltered, 0, sizeof(pktFiltered));
}

FFmpegDemuxer::~FFmpegDemuxer() {
	Close();
}

Error FFmpegDemuxer::Open(const char* szFilePath) {
	Close();
	AVFormatContext* ctx = nullptr;
	auto             err = CreateFormatContext(szFilePath, ctx);
	if (!err.OK())
		return err;
	return Open(ctx);
}

Error FFmpegDemuxer::Open(DataProvider* pDataProvider) {
	Close();
	AVFormatContext* ctx = nullptr;
	auto             err = CreateFormatContext(pDataProvider, ctx);
	if (!err.OK())
		return err;
	return Open(ctx);
}

Error FFmpegDemuxer::Open(AVFormatContext* _fmtc) {
	Close();
	fmtc = _fmtc;
	if (!fmtc)
		return Error("No AVFormatContext provided.");

	//LOG(INFO) << "Media format: " << fmtc->iformat->long_name << " (" << fmtc->iformat->name << ")";

	ck(avformat_find_stream_info(fmtc, NULL));
	iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	if (iVideoStream < 0)
		return Error("Could not find video stream in input file");

	//fmtc->streams[iVideoStream]->need_parsing = AVSTREAM_PARSE_NONE;
	eVideoCodec = fmtc->streams[iVideoStream]->codecpar->codec_id;
	nWidth      = fmtc->streams[iVideoStream]->codecpar->width;
	nHeight     = fmtc->streams[iVideoStream]->codecpar->height;
	nBitDepth   = 8;
	if (fmtc->streams[iVideoStream]->codecpar->format == AV_PIX_FMT_YUV420P10LE)
		nBitDepth = 10;
	if (fmtc->streams[iVideoStream]->codecpar->format == AV_PIX_FMT_YUV420P12LE)
		nBitDepth = 12;

	bMp4H264 = eVideoCodec == AV_CODEC_ID_H264 && (!strcmp(fmtc->iformat->long_name, "QuickTime / MOV") || !strcmp(fmtc->iformat->long_name, "FLV (Flash Video)") || !strcmp(fmtc->iformat->long_name, "Matroska / WebM"));

	timebase = fmtc->streams[iVideoStream]->time_base;

	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;
	av_init_packet(&pktFiltered);
	pktFiltered.data = NULL;
	pktFiltered.size = 0;

	if (bMp4H264) {
		const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
		if (!bsf)
			return Error("av_bsf_get_by_name(\"h264_mp4toannexb\") failed");
		ck(av_bsf_alloc(bsf, &bsfc));
		bsfc->par_in = fmtc->streams[iVideoStream]->codecpar;
		ck(av_bsf_init(bsfc));
	}

	return Error();
}

void FFmpegDemuxer::Close() {
	if (pkt.data) {
		av_packet_unref(&pkt);
		memset(&pkt, 0, sizeof(pkt));
	}
	if (pktFiltered.data) {
		av_packet_unref(&pktFiltered);
		memset(&pktFiltered, 0, sizeof(pktFiltered));
	}

	if (fmtc) {
		avformat_close_input(&fmtc);
		fmtc = nullptr;
	}
	if (avioc) {
		av_freep(&avioc->buffer);
		av_freep(&avioc);
		avioc = nullptr;
	}
}

Error FFmpegDemuxer::CreateFormatContext(DataProvider* pDataProvider, AVFormatContext*& ctx) {
	if (!AreCodecsRegistered) {
		AreCodecsRegistered = true;
		av_register_all();
	}

	if (!(ctx = avformat_alloc_context()))
		return Error("avformat_alloc_context failed");

	uint8_t* avioc_buffer      = NULL;
	int      avioc_buffer_size = 8 * 1024 * 1024;
	avioc_buffer               = (uint8_t*) av_malloc(avioc_buffer_size);
	if (!avioc_buffer)
		return Error::Fmt("Failed to allocate avioc_buffer of %v bytes", avioc_buffer_size);

	avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size, 0, pDataProvider, &ReadPacket, NULL, NULL);
	if (!avioc)
		return Error("avio_alloc_context failed");

	ctx->pb = avioc;

	return cuErr(avformat_open_input(&ctx, NULL, NULL, NULL));
}

Error FFmpegDemuxer::CreateFormatContext(const char* szFilePath, AVFormatContext*& ctx) {
	if (!AreCodecsRegistered) {
		AreCodecsRegistered = true;
		av_register_all();
	}
	avformat_network_init();

	int r = avformat_open_input(&ctx, szFilePath, NULL, NULL);
	if (r < 0)
		return IVideo::TranslateAvErr(r, tsf::fmt("Could not open video file %v", szFilePath).c_str());

	return Error();
}

bool FFmpegDemuxer::Demux(uint8_t** ppVideo, int* pnVideoBytes, int64_t* pts) {
	if (!fmtc) {
		return false;
	}

	*pnVideoBytes = 0;

	if (pkt.data) {
		av_packet_unref(&pkt);
	}

	int e = 0;
	while ((e = av_read_frame(fmtc, &pkt)) >= 0 && pkt.stream_index != iVideoStream) {
		av_packet_unref(&pkt);
	}
	if (e < 0) {
		return false;
	}

	if (bMp4H264) {
		if (pktFiltered.data) {
			av_packet_unref(&pktFiltered);
		}
		ck(av_bsf_send_packet(bsfc, &pkt));
		ck(av_bsf_receive_packet(bsfc, &pktFiltered));
		*ppVideo      = pktFiltered.data;
		*pnVideoBytes = pktFiltered.size;
		if (pts)
			*pts = pktFiltered.pts;
	} else {
		*ppVideo      = pkt.data;
		*pnVideoBytes = pkt.size;
		if (pts)
			*pts = pkt.pts;
	}

	return true;
}

} // namespace video
} // namespace imqs