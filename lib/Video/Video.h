#pragma once

extern "C" {
#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

#include <lib/pal/pal.h>
#include <lib/gfx/gfx.h>

#ifdef _WIN32
#define IMQS_VIDEO_API __declspec(dllimport)
#else
#define IMQS_VIDEO_API
#endif

#include "Decode.h"

#include "NVidia/NVVideo.h"