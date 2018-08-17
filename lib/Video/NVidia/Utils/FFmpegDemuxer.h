/*
* Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
}
#include "NvCodecUtils.h"
#include "../NvDecoder/nvcuvid.h"

class FFmpegDemuxer {
private:
	AVFormatContext* fmtc  = NULL;
	AVIOContext*     avioc = NULL;
	AVPacket         pkt, pktFiltered;
	AVBSFContext*    bsfc = NULL;

	int       iVideoStream;
	bool      bMp4H264;
	AVCodecID eVideoCodec;
	int       nWidth, nHeight, nBitDepth;

public:
	class DataProvider {
	public:
		virtual ~DataProvider() {}
		virtual int GetData(uint8_t* pBuf, int nBuf) = 0;
	};

private:
	FFmpegDemuxer(AVFormatContext* fmtc);
	AVFormatContext* CreateFormatContext(DataProvider* pDataProvider);
	AVFormatContext* CreateFormatContext(const char* szFilePath);

public:
	FFmpegDemuxer(const char* szFilePath) : FFmpegDemuxer(CreateFormatContext(szFilePath)) {}
	FFmpegDemuxer(DataProvider* pDataProvider) : FFmpegDemuxer(CreateFormatContext(pDataProvider)) {}
	~FFmpegDemuxer() {
		if (pkt.data) {
			av_packet_unref(&pkt);
		}
		if (pktFiltered.data) {
			av_packet_unref(&pktFiltered);
		}

		avformat_close_input(&fmtc);
		if (avioc) {
			av_freep(&avioc->buffer);
			av_freep(&avioc);
		}
	}
	AVCodecID GetVideoCodec() {
		return eVideoCodec;
	}
	int GetWidth() {
		return nWidth;
	}
	int GetHeight() {
		return nHeight;
	}
	int GetBitDepth() {
		return nBitDepth;
	}
	int GetFrameSize() {
		return nBitDepth == 8 ? nWidth * nHeight * 3 / 2 : nWidth * nHeight * 3;
	}
	bool Demux(uint8_t** ppVideo, int* pnVideoBytes);

	static int ReadPacket(void* opaque, uint8_t* pBuf, int nBuf) {
		return ((DataProvider*) opaque)->GetData(pBuf, nBuf);
	}
};

inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
	switch (id) {
	case AV_CODEC_ID_MPEG1VIDEO: return cudaVideoCodec_MPEG1;
	case AV_CODEC_ID_MPEG2VIDEO: return cudaVideoCodec_MPEG2;
	case AV_CODEC_ID_MPEG4: return cudaVideoCodec_MPEG4;
	case AV_CODEC_ID_VC1: return cudaVideoCodec_VC1;
	case AV_CODEC_ID_H264: return cudaVideoCodec_H264;
	case AV_CODEC_ID_HEVC: return cudaVideoCodec_HEVC;
	case AV_CODEC_ID_VP8: return cudaVideoCodec_VP8;
	case AV_CODEC_ID_VP9: return cudaVideoCodec_VP9;
	case AV_CODEC_ID_MJPEG: return cudaVideoCodec_JPEG;
	default: return cudaVideoCodec_NumCodecs;
	}
}
