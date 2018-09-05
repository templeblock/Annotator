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

#define VIDEOHANDLE_DEFINED
typedef imqs::video::IVideo* VideoHandle;
#include "CVideo.h"

// This is a C interface to IVideo. It was created so that we can consume this library from Python, with CFFI

using namespace imqs;
using namespace imqs::video;
using namespace imqs::gfx;

#define MAKE_ERR(e) strdup(e.Message())

extern "C" {

char* OpenVideoFile(const char* driver, const char* filename, VideoHandle* handle) {
	imqs::video::IVideo* v = nullptr;
	if (driver == nullptr || driver[0] == 0)
		v = new PlatformVideoType();
	else if (strcmp(driver, "nvidia") == 0)
		v = new PlatformVideoType();
	else if (strcmp(driver, "ffmpeg") == 0)
		v = new imqs::video::VideoFile();
	else
		return strdup(tsf::fmt("Unknown driver %v. Valid drivers are nvidia, ffmpeg, or null for default", driver).c_str());

	auto e = v->OpenFile(filename);
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

void VideoInfo(VideoHandle v, int* width, int* height, double* durationSeconds) {
	int64_t micros;
	v->Info(*width, *height, micros);
	*durationSeconds = (double) micros / 1000000.0;
}

char* DecodeFrameRGBA(VideoHandle v, int width, int height, void* buf, int stride, double* timeSeconds) {
	auto e = v->DecodeFrameRGBA(width, height, buf, stride, timeSeconds);
	if (!e.OK())
		return MAKE_ERR(e);
	return nullptr;
}

char* VideoSeek(VideoHandle v, double timeSeconds, unsigned seekFlag) {
	unsigned f = 0;
	if (!!(seekFlag & VideoSeekFlagAny))
		f |= imqs::video::SeekFlagAny;
	auto e = v->SeekToMicrosecond(int64_t(timeSeconds * 1000000.0), f);
	if (!e.OK())
		return MAKE_ERR(e);
	return nullptr;
}
}