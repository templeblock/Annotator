#ifndef CVIDEO_H_INCLUDED
#define CVIDEO_H_INCLUDED

typedef void* VideoHandle;

char* OpenVideoFile(const char* filename, VideoHandle* handle);
void  CloseVideo(VideoHandle v);
void  VideoInfo(VideoHandle v, int* width, int* height);
char* DecodeFrameRGBA(VideoHandle v, int width, int height, void* buf, int stride, double* timeSeconds);

#endif // CVIDEO_H_INCLUDED