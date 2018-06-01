#pragma once

#include <lib/gfx/gfx.h>
#include <lib/Video/Video.h>

#ifdef _WIN32
#define IMQS_TRAIN_API __declspec(dllexport)
#else
#define IMQS_TRAIN_API
#endif
