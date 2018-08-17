#include "pch.h"
#include "FFmpegDemuxer.h"

FFmpegDemuxer::FFmpegDemuxer(AVFormatContext* fmtc) : fmtc(fmtc) {
	if (!fmtc) {
		LOG(ERROR) << "No AVFormatContext provided.";
		return;
	}

	LOG(INFO) << "Media format: " << fmtc->iformat->long_name << " (" << fmtc->iformat->name << ")";

	ck(avformat_find_stream_info(fmtc, NULL));
	iVideoStream = av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
	if (iVideoStream < 0) {
		LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " "
		           << "Could not find stream in input file";
		return;
	}

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

	av_init_packet(&pkt);
	pkt.data = NULL;
	pkt.size = 0;
	av_init_packet(&pktFiltered);
	pktFiltered.data = NULL;
	pktFiltered.size = 0;

	if (bMp4H264) {
		const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
		if (!bsf) {
			LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__ << " "
			           << "av_bsf_get_by_name() failed";
			return;
		}
		ck(av_bsf_alloc(bsf, &bsfc));
		bsfc->par_in = fmtc->streams[iVideoStream]->codecpar;
		ck(av_bsf_init(bsfc));
	}
}

AVFormatContext* FFmpegDemuxer::CreateFormatContext(DataProvider* pDataProvider) {
	av_register_all();

	AVFormatContext* ctx = NULL;
	if (!(ctx = avformat_alloc_context())) {
		LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
		return NULL;
	}

	uint8_t* avioc_buffer      = NULL;
	int      avioc_buffer_size = 8 * 1024 * 1024;
	avioc_buffer               = (uint8_t*) av_malloc(avioc_buffer_size);
	if (!avioc_buffer) {
		LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
		return NULL;
	}
	avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size,
	                           0, pDataProvider, &ReadPacket, NULL, NULL);
	if (!avioc) {
		LOG(ERROR) << "FFmpeg error: " << __FILE__ << " " << __LINE__;
		return NULL;
	}
	ctx->pb = avioc;

	ck(avformat_open_input(&ctx, NULL, NULL, NULL));
	return ctx;
}

AVFormatContext* FFmpegDemuxer::CreateFormatContext(const char* szFilePath) {
	av_register_all();
	avformat_network_init();

	AVFormatContext* ctx = NULL;
	ck(avformat_open_input(&ctx, szFilePath, NULL, NULL));
	return ctx;
}

bool FFmpegDemuxer::Demux(uint8_t** ppVideo, int* pnVideoBytes) {
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
	} else {
		*ppVideo      = pkt.data;
		*pnVideoBytes = pkt.size;
	}

	return true;
}
