#include "pch.h"
#include "IVideo.h"
#include "Decode.h"

#ifdef _WIN32
#include "NVVideoStub_windows.h"
typedef imqs::video::VideoFile PlatformVideoType;
#else
#include "NVidia_linux/NVVideo.h"
typedef imqs::video::NVVideo PlatformVideoType;
#endif

// This is a C interface to IVideo. It was created so that we can consume this library from Python, with CFFI

using namespace imqs;
using namespace imqs::video;
using namespace imqs::gfx;

#define MAKE_ERR(e) strdup(e.Message())

namespace imqs {
namespace video {
} // namespace video
} // namespace imqs

extern "C" {

//typedef void* VideoHandle;
typedef imqs::video::IVideo* VideoHandle;

char* OpenVideoFile(const char* filename, VideoHandle* handle) {
	PlatformVideoType* v = new PlatformVideoType();
	auto               e = v->OpenFile(filename);
	if (!e.OK()) {
		delete v;
		return MAKE_ERR(e);
	}
	*handle = v;
	return nullptr;
}

void CloseVideo(VideoHandle v) {
	delete v;
}

void VideoInfo(VideoHandle v, int* width, int* height) {
	v->Info(*width, *height);
}

char* DecodeFrameRGBA(VideoHandle v, int width, int height, void* buf, int stride, double* timeSeconds) {
	auto e = v->DecodeFrameRGBA(width, height, buf, stride, timeSeconds);
	if (!e.OK())
		return MAKE_ERR(e);
	return nullptr;
}
}