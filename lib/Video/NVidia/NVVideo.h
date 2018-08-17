#pragma once

#include "NvDecoder/NvDecoder.h"
#include "Utils/FFmpegDemuxer.h"

namespace imqs {
namespace roadproc {

/* Use NvDecoder to decode video to OpenGL
This code was built up from the AppDecGL sample in the NVidia Video SDK, version 8.2.15
*/
class NVVideo {
public:
	enum DecodeStates {
		DecodeStateNotStarted = 0,
		DecodeStateRunning,
		DecodeStateFinished,
	};
	NVVideo();
	~NVVideo();
	Error Initialize(int iGPU = 0);
	Error OpenFile(std::string filename);
	int   Width();
	int   Height();
	Error DecodeFrameRGBA(int width, int height, void* buf, int stride);

private:
	struct HostFrame {
		gfx::Image Img;
		int64_t    TimeNano = 0;
	};
	CUcontext                CUCtx            = nullptr;
	FFmpegDemuxer*           Demuxer          = nullptr;
	NvDecoder*               Decoder          = nullptr;
	int                      DeviceBufferSize = 20;
	int                      HostBufferSize   = 5;
	size_t                   FrameSize        = 0;
	std::vector<CUdeviceptr> DeviceFrames;
	HostFrame*               HostFrames;
	std::atomic<int>         HostHead;
	std::atomic<int>         HostTail;
	Semaphore                SemDecode; // Incremented when a frame is posted to the HostFrames ring buffer
	std::atomic<int>         DecodeState;
	std::thread              DecodeThread;
	std::atomic<int>         ExitSignaled;

	Error InitBuffers();
	void  DecodeThreadFunc();
};

} // namespace roadproc
} // namespace imqs