#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#define _CRT_SECURE_NO_WARNINGS 1

#include <pal/pal.h>

extern "C" {
#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#ifdef _MSC_VER
#define IMQS_VIDEO_API __declspec(dllexport)
#else
#define IMQS_VIDEO_API
#endif
