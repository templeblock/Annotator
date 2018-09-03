#pragma once

#include "NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"
#include "../IVideo.h"

namespace imqs {
namespace video {

/* Use NvDecoder to decode video to OpenGL
This code was built up from the AppDecGL sample in the NVidia Video SDK, version 8.2.15
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
	Error Initialize(int iGPU = 0);
	void  Close();
	int   Width();
	int   Height();

	// IVideo
	Error OpenFile(std::string filename) override;
	void  Info(int& width, int& height) override;
	Error DecodeFrameRGBA(int width, int height, void* buf, int stride, double* timeSeconds = nullptr) override;
	Error SeekToMicrosecond(int64_t microsecond, unsigned flags = Seek::None) override;

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
	CUcontext     CUCtx = nullptr;
	FFmpegDemuxer Demuxer;
	NvDecoder*    Decoder          = nullptr;
	int           DeviceBufferSize = 20;
	int           HostBufferSize   = 5;
	size_t        FrameSize        = 0;

	std::vector<DeviceFrame> DeviceFrames;
	std::atomic<int>         DeviceHead;
	std::atomic<int>         DeviceTail;
	Semaphore*               SemDeviceFramesReady = nullptr; // Number of occupied slots in DeviceFrames
	Semaphore*               SemDeviceFramesFree  = nullptr; // Number of available slots in DeviceFrames

	// NOTE: If the dual semaphore approach works for DeviceFrames, then use it for the host frames too

	std::vector<HostFrame> HostFrames;
	std::atomic<int>       HostHead;
	std::atomic<int>       HostTail;
	Semaphore*             SemHostFramesReady = nullptr; // Number of occupied slots in HostFrames

	std::atomic<int> DecodeState;
	std::thread      DecodeThread;
	std::atomic<int> ExitSignaled;

	Error InitBuffers();
	void  DecodeThreadFunc();
};

} // namespace video
} // namespace imqs