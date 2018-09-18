#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
}
#include "NvCodecUtils.h"
#include "ImqsUtils.h"
#include "../NvDecoder/nvcuvid.h"

namespace imqs {
namespace video {

class FFmpegDemuxer {
private:
	AVFormatContext* fmtc  = nullptr;
	AVIOContext*     avioc = nullptr;
	AVPacket         pkt;
	AVPacket         pktFiltered;
	AVBSFContext*    bsfc = nullptr;
	AVRational       timebase;

	int       iVideoStream = -1;
	bool      bMp4H264     = false;
	AVCodecID eVideoCodec  = AV_CODEC_ID_NONE;
	int       nWidth       = 0;
	int       nHeight      = 0;
	int       nBitDepth    = 0;
	int64_t   Duration     = 0;

public:
	class DataProvider {
	public:
		virtual ~DataProvider() {}
		virtual int GetData(uint8_t* pBuf, int nBuf) = 0;
	};

private:
	Error CreateFormatContext(DataProvider* pDataProvider, AVFormatContext*& ctx);
	Error CreateFormatContext(const char* szFilePath, AVFormatContext*& ctx);

public:
	FFmpegDemuxer();
	~FFmpegDemuxer();

	Error Open(AVFormatContext* fmtc);
	Error Open(const char* szFilePath);
	Error Open(DataProvider* pDataProvider);
	void  Close();

	AVCodecID  GetVideoCodec() { return eVideoCodec; }
	int        GetWidth() { return nWidth; }
	int        GetHeight() { return nHeight; }
	int        GetBitDepth() { return nBitDepth; }
	int        GetFrameSize() { return nBitDepth == 8 ? nWidth * nHeight * 3 / 2 : nWidth * nHeight * 3; }
	AVRational GetTimebase() { return timebase; }
	double     GetDurationSeconds();

	Error SeekPts(int64_t pts);
	bool  Demux(uint8_t** ppVideo, int* pnVideoBytes, int64_t* pts);

	int64_t       SecondsToPts(double s) { return int64_t(s / av_q2d(timebase)); }
	double        PtsToSeconds(int64_t pts) { return PtsToSeconds(pts, timebase); }
	static double PtsToSeconds(int64_t pts, AVRational timebase) { return av_q2d(av_mul_q({(int) pts, 1}, timebase)); }

	static int ReadPacket(void* opaque, uint8_t* pBuf, int nBuf) { return ((DataProvider*) opaque)->GetData(pBuf, nBuf); }
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

} // namespace video
} // namespace imqs