#include "pch.h"
//#include <iostream>
#include "NVVideo.h"
#include "Utils/NvCodecUtils.h"
#include "Utils/ImqsUtils.h"
//#include <cuda_runtime_api.h>
//#include "FramePresenterGL.h"
//#include "../Common/AppDecUtils.h"

simplelogger::Logger* logger = simplelogger::LoggerFactory::CreateConsoleLogger();
using namespace std;

namespace imqs {
namespace video {

CUcontext CUCtx = nullptr;

NVVideo::NVVideo() {
	HostHead     = 0;
	HostTail     = 0;
	DeviceTail   = 0;
	DeviceHead   = 0;
	DecodeState  = DecodeStateNotStarted;
	ExitSignaled = 0;
}

NVVideo::~NVVideo() {
	Close();
}

Error NVVideo::Initialize(int iGPU) {
	IMQS_ASSERT(!CUCtx);

	auto err = cuErr(cuInit(0));
	if (!err.OK())
		return err;
	int nGPU = 0;
	err      = cuErr(cuDeviceGetCount(&nGPU));
	if (iGPU < 0 || iGPU >= nGPU)
		return Error::Fmt("NVVideo: Invalid GPU specified (%v). Only %v GPUs found", iGPU, nGPU);
	CUdevice cuDevice = 0;
	err |= cuErr(cuDeviceGet(&cuDevice, iGPU));
	char szDeviceName[80];
	err |= cuErr(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
	//tsf::print("GPU in use: %v\n", szDeviceName);
	err = cuErr(cuCtxCreate(&CUCtx, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));
	if (!err.OK())
		return err;

	return Error();
}

void NVVideo::Shutdown() {
	if (CUCtx)
		cuCtxDestroy(CUCtx);
}

Error NVVideo::OpenFile(std::string filename) {
	Close();
	if (!CUCtx) {
		auto err = Initialize();
		if (!err.OK())
			return err;
	}
	auto err = Demuxer.Open(filename.c_str());
	if (!err.OK())
		return err;
	Filename = filename;
	Decoder  = new NvDecoder(CUCtx, Demuxer.GetWidth(), Demuxer.GetHeight(), true, FFmpeg2NvCodecId(Demuxer.GetVideoCodec()));
	return Error();
}

Error NVVideo::CloseAndReopen(bool seekToLastPTS) {
	Close();
	auto err = OpenFile(Filename);
	if (!err.OK())
		return err;

	if (seekToLastPTS)
		return Demuxer.SeekPts(LastObservedPTS);
	else
		return Error();
}

void NVVideo::Info(int& width, int& height, int64_t& durationMicroseconds) {
	width                = Width();
	height               = Height();
	durationMicroseconds = Demuxer.GetDurationSeconds() * 1000000;
}

Error NVVideo::SetOutputResolution(int width, int height) {
	if (width % 2 != 0 || height % 2 != 0)
		return Error("Output size must be a multiple of 2 (due to the way our 4:2:0 decoding functions are written)");

	if (DecodeState == DecodeStateRunning) {
		auto err = CloseAndReopen(true);
		if (!err.OK())
			return err;
	}
	OutputWidth  = width;
	OutputHeight = height;
	// wait for the decode thread to flush all stale buffers. If we don't do this, then this thread
	// will go ahead and ask for more frames, but the system is not in a consistent state, until
	// OutSideDirty = 0 again.
	//while (DecodeState == DecodeStateRunning && OutSizeDirty == 1) {
	//	os::Sleep(time::Millisecond);
	//}
	return Error();
}

void NVVideo::Close() {
	ExitSignaled = 1;
	if (SemHostFramesFree) {
		// Unblock the decoding thread
		SemHostFramesFree->signal();
	}
	if (DecodeThread.joinable())
		DecodeThread.join();

	for (auto& f : HostFrames)
		cudaFreeHost(f.Img.Data);
	HostFrames.resize(0);

	for (auto f : DeviceFrames)
		cudaFree(f.Frame);
	DeviceFrames.resize(0);

	cudaFree(DeviceTempFrame);
	DeviceTempFrame      = nullptr;
	DeviceTempFramePitch = 0;

	FrameSize    = 0;
	ExitSignaled = 0;
	HostHead     = 0;
	HostTail     = 0;
	DeviceHead   = 0;
	DeviceTail   = 0;
	delete SemHostFramesFree;
	SemHostFramesFree = nullptr;
	delete SemHostFramesReady;
	SemHostFramesReady = nullptr;
	delete SemDeviceFramesFree;
	SemDeviceFramesFree = nullptr;
	delete SemDeviceFramesReady;
	SemDeviceFramesReady = nullptr;
	DecodeState          = DecodeStateNotStarted;
}

int NVVideo::Width() {
	return Demuxer.GetWidth();
}

int NVVideo::Height() {
	return Demuxer.GetHeight();
}

int NVVideo::OutWidth() {
	return OutputWidth ? OutputWidth : Width();
}

int NVVideo::OutHeight() {
	return OutputHeight ? OutputHeight : Height();
}

Error NVVideo::DecodeFramePrelude() {
	if (FrameSize == 0) {
		auto err = InitBuffers();
		if (!err.OK())
			return err;
	} else {
		// If you're going to call SetOutputResolution(), then you must do so before decoding any frames
		IMQS_ASSERT(FrameSize == OutWidth() * 4 * OutHeight());
	}

	if (DecodeState == DecodeStateNotStarted) {
		DecodeState  = DecodeStateRunning;
		DecodeThread = std::thread([&]() -> void {
			DecodeThreadFunc();
		});
	}

	return Error();
}

Error NVVideo::DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds) {
	IMQS_ASSERT(OutputMode == OutputCPU);

	auto err = DecodeFramePrelude();
	if (!err.OK())
		return err;

	//if (DecodeState == DecodeStateFinished && HostTail == HostHead) {
	//	// The decoder has consumed all frames of the video, and we've consumed them all
	//	return ErrEOF;
	//}

	SemHostFramesReady->wait();
	if (HostTail == HostHead) {
		// The decoder has finished all frames, and exited. It signaled the semaphore one last time to wake us up.
		IMQS_ASSERT(DecodeState == DecodeStateFinished);
		return ErrEOF;
	}

	IMQS_ASSERT(OutWidth() == width);
	IMQS_ASSERT(OutHeight() == height);
	IMQS_ASSERT(HostTail < HostHead);

	HostFrame& f         = HostFrames[HostTail % HostBufferSize];
	auto       lineBytes = f.Img.BytesPerLine();
	auto       srcStride = f.Img.Stride;
	uint8_t*   src       = (uint8_t*) f.Img.Data;
	uint8_t*   dst       = (uint8_t*) buf;
	for (int y = 0; y < height; y++)
		memcpy(dst + y * stride, src + y * srcStride, lineBytes);

	if (timeSeconds)
		*timeSeconds = Demuxer.PtsToSeconds(f.Pts);
	LastObservedPTS = f.Pts;

	HostTail++;
	SemHostFramesFree->signal();

	return Error();
}

Error NVVideo::DecodeFrameRGBA_GPU(CudaFrame& frame) {
	IMQS_ASSERT(OutputMode == OutputGPU);

	auto err = DecodeFramePrelude();
	if (!err.OK())
		return err;

	SemDeviceFramesReady->wait();
	if (DeviceTail == DeviceHead) {
		// The decoder has finished all frames, and exited. It signaled the semaphore one last time to wake us up.
		IMQS_ASSERT(DecodeState == DecodeStateFinished);
		return ErrEOF;
	}

	IMQS_ASSERT(DeviceTail < DeviceHead);

	DeviceFrame& f   = DeviceFrames[DeviceTail % DeviceBufferSize];
	frame.Frame      = f.Frame;
	frame.Stride     = f.Stride;
	frame.Pts        = f.Pts;
	frame.PtsSeconds = Demuxer.PtsToSeconds(f.Pts);

	LastObservedPTS = f.Pts;

	// The consumer now owns the pointer, so we need to allocate more memory for the next frame
	// This is potentially expensive - these frequent re-allocs. The only numbers I could find for
	// cudaMalloc overhead are old (2011 IIRC), so I'm not sure what the cost of this is today.
	f.Frame = nullptr;
	err     = cuErr(cudaMalloc(&f.Frame, FrameSize));
	if (!err.OK())
		return err;

	DeviceTail++;
	SemDeviceFramesFree->signal();

	return Error();
}

Error NVVideo::SeekToMicrosecond(int64_t microsecond, unsigned flags) {
	if (DecodeState == DecodeStateRunning) {
		auto err = CloseAndReopen(false);
		if (!err.OK())
			return err;
	}
	return Demuxer.SeekPts(Demuxer.SecondsToPts(microsecond / 1000000.0));
}

Error NVVideo::InitBuffers() {
	DecodeState = DecodeStateNotStarted;

	FrameSize = OutWidth() * OutHeight() * 4;

	IMQS_ASSERT(HostFrames.size() == 0);
	IMQS_ASSERT(DeviceFrames.size() == 0);

	HostFrames.resize(HostBufferSize);
	for (int i = 0; i < HostBufferSize; i++) {
		void* b;
		auto  err = cuErr(cudaMallocHost(&b, FrameSize));
		if (!err.OK())
			return err;
		HostFrames[i].Img = gfx::Image(gfx::ImageFormat::RGBA, gfx::Image::ConstructWindow, OutWidth() * 4, b, OutWidth(), OutHeight());
	}
	SemHostFramesReady = new Semaphore();
	SemHostFramesFree  = new Semaphore();
	SemHostFramesFree->signal(HostFrames.size());
	HostHead = 0;
	HostTail = 0;

	DeviceFrames.resize(DeviceBufferSize);
	for (int i = 0; i < DeviceBufferSize; i++) {
		auto err = cuErr(cudaMalloc(&DeviceFrames[i].Frame, FrameSize));
		if (!err.OK())
			return err;
	}
	SemDeviceFramesReady = new Semaphore();
	SemDeviceFramesFree  = new Semaphore();
	SemDeviceFramesFree->signal(DeviceFrames.size());

	return Error();
}

// We can't do this until we've decoded at least one frame, because only then do we know the bit depth.
Error NVVideo::AllocResizeBuffer() {
	if (DeviceTempFrame)
		return Error();

	if (OutWidth() != Width() || OutHeight() != Height()) {
		// This is a requirement for NV12 encoding (4:2:0) - specifically the CUDA code that does resizing and color conversion
		IMQS_ASSERT(OutWidth() % 2 == 0);
		IMQS_ASSERT(OutHeight() % 2 == 0);

		if (Decoder->GetBitDepth() == 8)
			DeviceTempFramePitch = OutWidth(); // Nv12 (planar 4:2:0 8-bit)
		else
			DeviceTempFramePitch = OutWidth() * 2; // P016 (planar 4:2:0 16-bit)

		size_t bytes = OutHeight() * DeviceTempFramePitch * 3 / 2;
		auto   err   = cuErr(cudaMalloc(&DeviceTempFrame, bytes));
		if (!err.OK())
			return err;
	}
	return Error();
}

void NVVideo::DecodeThreadFunc() {
	int nFrame = 0;

	auto sendFramesToHost = [&]() {
		while (DeviceTail != DeviceHead) {
			/*
			for (int spin = 0; HostHead - HostTail == HostBufferSize; spin++) {
				// Obviously sleeping is never great, but I don't know how to do this correctly. I just feel
				// like there is some kind of deadlock going on if we also use a semaphore from the reading
				// side. I'm sure there is an elegant solution to this problem.
				int sleepMicros = std::min(100 * spin, 10 * 1000); // max sleep = 10 milliseconds
				os::Sleep(time::Microsecond * sleepMicros);
				if (ExitSignaled)
					break;
			}
			*/
			SemHostFramesFree->wait(); // we expect to block on this (waiting for 'main' thread to consume frames)
			if (ExitSignaled)
				break;
			SemDeviceFramesReady->wait(); // should never block on this
			auto& dFrame = DeviceFrames[DeviceTail % DeviceBufferSize];
			auto& hFrame = HostFrames[HostHead % HostBufferSize];
			cudaMemcpy(hFrame.Img.Data, (const void*) dFrame.Frame, FrameSize, cudaMemcpyDeviceToHost);
			hFrame.Pts = dFrame.Pts;
			HostHead++;
			SemHostFramesReady->signal();
			DeviceTail++;
			SemDeviceFramesFree->signal();
		}
	};

	int maxNFrames = 0;
	while (!ExitSignaled) {
		int64_t*  timeStamps;
		int       videoBytes     = 0;
		int       nFrameReturned = 0;
		uint8_t*  pVideo         = nullptr;
		uint8_t** ppFrame;
		int64_t   pts = 0;
		bool      ok  = Demuxer.Demux(&pVideo, &videoBytes, &pts);
		if (!ok)
			tsf::print("Demuxer.Demux failed!\n");
		ok = Decoder->Decode(pVideo, videoBytes, &ppFrame, &nFrameReturned, 0, &timeStamps, pts);
		if (!ok)
			tsf::print("Decoder->Decode failed!\n");
		//if (!nFrame && nFrameReturned)
		//	LOG(INFO) << dec.GetVideoInfo();

		// The maximum number I've seen here is 13.
		//int prevMax = maxNFrames;
		//maxNFrames  = max(maxNFrames, nFrameReturned);
		//if (maxNFrames > prevMax) {
		//	tsf::print("max nframes: %v\n", maxNFrames);
		//}

		if (OutputMode == OutputCPU) {
			if (nFrameReturned + (DeviceHead - DeviceTail) > DeviceBufferSize) {
				// if we cannot fit nFrameReturned into our device buffer, then flush all remaining device frames to host
				//tsf::print("too much!\n");
				sendFramesToHost();
			}
			IMQS_ASSERT(nFrameReturned + (DeviceHead - DeviceTail) <= DeviceBufferSize);
		}

		for (int i = 0; i < nFrameReturned; i++) {
			SemDeviceFramesFree->wait();
			void* resizedFrame      = ppFrame[i];
			int   resizedFramePitch = Decoder->GetWidth();
			auto  err               = AllocResizeBuffer();
			if (!err.OK())
				IMQS_DIE_MSG(tsf::fmt("Error allocating image resize buffer: %v", err.Message()).c_str());
			if (OutWidth() != Width() || OutHeight() != Height()) {
				if (Decoder->GetBitDepth() == 8) {
					ResizeNv12((uint8_t*) DeviceTempFrame, DeviceTempFramePitch, OutWidth(), OutHeight(), (uint8_t*) ppFrame[i], Decoder->GetWidth(), Decoder->GetWidth(), Decoder->GetHeight());
				} else {
					ResizeP016((uint8_t*) DeviceTempFrame, DeviceTempFramePitch, OutWidth(), OutHeight(), (uint8_t*) ppFrame[i], 2 * Decoder->GetWidth(), Decoder->GetWidth(), Decoder->GetHeight());
				}
				resizedFrame      = DeviceTempFrame;
				resizedFramePitch = DeviceTempFramePitch;
			}
			auto& dFrame  = DeviceFrames[DeviceHead % DeviceBufferSize];
			dFrame.Stride = OutWidth() * 4;
			if (Decoder->GetBitDepth() == 8)
				Nv12ToRgba32((uint8_t*) resizedFrame, resizedFramePitch, (uint8_t*) dFrame.Frame, dFrame.Stride, OutWidth(), OutHeight());
			else
				P016ToRgba32((uint8_t*) resizedFrame, 2 * resizedFramePitch, (uint8_t*) dFrame.Frame, dFrame.Stride, OutWidth(), OutHeight());
			dFrame.Pts = timeStamps[i];
			DeviceHead++;
			SemDeviceFramesReady->signal();
		}

		if (OutputMode == OutputCPU) {
			// we want to flush frames before we hit our limit, otherwise the reader thread will stall, while waiting for frames from us
			if (DeviceHead - DeviceTail >= DeviceBufferSize - 5)
				sendFramesToHost();
		}

		nFrame += nFrameReturned;
		if (videoBytes == 0)
			break;
	};

	DecodeState = DecodeStateFinished;
	// wake up thread that's waiting for a result from us (inside DecodeFrameRGBA or DecodeFrameRGBA_GPU)
	SemHostFramesReady->signal();
	SemDeviceFramesReady->signal();
}

} // namespace video
} // namespace imqs

/*
int Decode(CUcontext cuContext, char* szInFilePath) {
	FFmpegDemuxer    demuxer(szInFilePath);
	NvDecoder        dec(cuContext, demuxer.GetWidth(), demuxer.GetHeight(), true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
	FramePresenterGL presenter(cuContext, demuxer.GetWidth(), demuxer.GetHeight());
	uint8_t*         dpFrame     = 0;
	int              nPitch      = 0;
	int              nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
	uint8_t*         pVideo = NULL;
	uint8_t**        ppFrame;
	do {
		demuxer.Demux(&pVideo, &nVideoBytes);
		dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
		if (!nFrame && nFrameReturned)
			LOG(INFO) << dec.GetVideoInfo();

		for (int i = 0; i < nFrameReturned; i++) {
			presenter.GetDeviceFrameBuffer(&dpFrame, &nPitch);
			if (dec.GetBitDepth() == 8)
				Nv12ToBgra32((uint8_t*) ppFrame[i], dec.GetWidth(), (uint8_t*) dpFrame, nPitch, dec.GetWidth(), dec.GetHeight());
			else
				P016ToBgra32((uint8_t*) ppFrame[i], 2 * dec.GetWidth(), (uint8_t*) dpFrame, nPitch, dec.GetWidth(), dec.GetHeight());
		}
		nFrame += nFrameReturned;
	} while (nVideoBytes);
	std::cout << "Total frame decoded: " << nFrame << std::endl;
	return 0;
}
*/

/**
*  This sample application illustrates the decoding of media file and display of decoded frames
*  in a window. This is done by CUDA interop with OpenGL.
*/
/*
int main(int argc, char** argv) {
	char szInFilePath[256] = "";
	int  iGpu              = 0;
	try {
		ParseCommandLine(argc, argv, szInFilePath, NULL, iGpu, NULL, NULL);
		CheckInputFile(szInFilePath);

		ck(cuInit(0));
		int nGpu = 0;
		ck(cuDeviceGetCount(&nGpu));
		if (iGpu < 0 || iGpu >= nGpu) {
			std::ostringstream err;
			err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
			throw std::invalid_argument(err.str());
		}
		CUdevice cuDevice = 0;
		ck(cuDeviceGet(&cuDevice, iGpu));
		char szDeviceName[80];
		ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
		std::cout << "GPU in use: " << szDeviceName << std::endl;
		CUcontext cuContext = NULL;
		ck(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));

		std::cout << "Decode with NvDecoder." << std::endl;
		Decode(cuContext, szInFilePath);
	} catch (const std::exception& ex) {
		std::cout << ex.what();
		exit(1);
	}
	return 0;
}
*/
