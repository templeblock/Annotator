#pragma once

#include "NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"
#include "../IVideo.h"

namespace imqs {
namespace video {

/* Use NvDecoder to decode video to OpenGL
This code was built up from the AppDecGL sample in the NVidia Video SDK, version 8.2.15

If you call Seek or SetOutputSize after decoding has started, then the system will:
1. Close everything
2. Reopen everything
3. Seek to the last observed PTS
4. Continue from there

The seeking to last observed PTS might not be 100% perfect, but in my observations of videos
from a Fuji X-T2, it seems much better than I would have thought. Generally, zero skipped
frames.

Closing and reopening everything is not the most efficient way to do this, but for our use
case it's perfectly fine. My initial approaches were to try and do things without closing
and reopening, but everything is much MUCH simpler to get right if we do it that way.

I have run this under cuda-memcheck, and I get a zillion memory violations inside
YuvToRgbKernel. However, everything appears to work fine. I haven't tried to run the
nvidia code unchanged through cuda-memcheck, so I don't know where the issue is.

NOTE. This system is not theoretically optimal. To be optimal, you must run all of the decode
phases on different threads. The docs for the NVidia Video SDK describe this. However,
our bottleneck is not the video decoding, so that's why this is not a priority.
*/
class NVVideo : public IVideo {
public:
	enum DecodeStates {
		DecodeStateNotStarted = 0,
		DecodeStateRunning,
		DecodeStateFinished,
	};
	enum OutputModes {
		OutputCPU, // Use DecodeFrameRGBA
		OutputGPU, // Use DecodeFrameRGBA_GPU
	};

	struct CudaFrame {
		void*   Frame      = nullptr;
		int     Stride     = 0;
		int64_t Pts        = 0;
		double  PtsSeconds = 0;
	};

	OutputModes OutputMode = OutputCPU; // Set this before opening the file

	NVVideo();
	~NVVideo() override;

	static Error Initialize(int iGPU = 0);
	static void  Shutdown();

	void Close();
	int  Width();
	int  Height();
	int  OutWidth();
	int  OutHeight();

	// IVideo
	Error OpenFile(std::string filename) override;
	void  Info(int& width, int& height, int64_t& durationMicroseconds) override;
	Error SetOutputResolution(int width, int height) override;
	Error DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds = nullptr) override;
	Error SeekToMicrosecond(int64_t microsecond, unsigned flags = SeekFlagNone) override;

	// Returns a pointer to CUDA memory, containing the next decoded RGBA video frame (and it's stride)
	// OutputMode must be set to OutputGPU before opening the file.
	// The pointer that is returned is now yours. You can do what you want with it, and you are responsible
	// for freeing it.
	Error DecodeFrameRGBA_GPU(CudaFrame& frame);

private:
	struct DeviceFrame {
		void*   Frame  = nullptr;
		int     Stride = 0;
		int64_t Pts    = 0;
	};
	struct HostFrame {
		gfx::Image Img; // The memory inside this img is special pinned memory - allocated with cudaMallocHost
		int64_t    Pts = 0;
	};
	std::string   Filename;
	FFmpegDemuxer Demuxer;
	NvDecoder*    Decoder          = nullptr;
	int           DeviceBufferSize = 20;
	int           HostBufferSize   = 5;
	size_t        FrameSize        = 0;
	int           OutputWidth      = 0;
	int           OutputHeight     = 0;
	int64_t       LastObservedPTS  = 0; // The PTS of the last frame that the user saw

	void*                    DeviceTempFrame      = nullptr; // If output size is different to native frame size, then this is an intermediate buffer for resizing
	int                      DeviceTempFramePitch = 0;
	std::vector<DeviceFrame> DeviceFrames;
	std::atomic<int>         DeviceHead;
	std::atomic<int>         DeviceTail;
	Semaphore*               SemDeviceFramesReady = nullptr; // Number of occupied slots in DeviceFrames
	Semaphore*               SemDeviceFramesFree  = nullptr; // Number of available slots in DeviceFrames

	std::vector<HostFrame> HostFrames;
	std::atomic<int>       HostHead;
	std::atomic<int>       HostTail;
	Semaphore*             SemHostFramesReady = nullptr; // Number of occupied slots in HostFrames
	Semaphore*             SemHostFramesFree  = nullptr; // Number of available slots in HostFrames

	std::atomic<int> DecodeState;
	std::thread      DecodeThread;
	std::atomic<int> ExitSignaled;

	Error CloseAndReopen(bool seekToLastPTS);
	Error AllocResizeBuffer();
	Error DecodeFramePrelude();
	Error InitBuffers();
	void  DecodeThreadFunc();
};

} // namespace video
} // namespace imqs