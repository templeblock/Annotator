#ifndef CVIDEO_H_INCLUDED
#define CVIDEO_H_INCLUDED

#ifndef VIDEOHANDLE_DEFINED
typedef void* VideoHandle;
#endif

enum VideoSeekFlags {
	VideoSeekFlagAny = 1, // seek to any frame, even non-keyframes
};

#ifdef __cplusplus
extern "C" {
#endif

char* OpenVideoFile(const char* driver, const char* filename, VideoHandle* handle);
void  CloseVideo(VideoHandle v);
void  VideoInfo(VideoHandle v, int* width, int* height, double* durationSeconds);
char* DecodeFrameRGBA(VideoHandle v, int width, int height, void* buf, int stride, double* timeSeconds);
char* VideoSeek(VideoHandle v, double timeSeconds, unsigned seekFlags);
void  SetOutputResolution(VideoHandle v, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // CVIDEO_H_INCLUDED