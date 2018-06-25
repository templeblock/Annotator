#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#define _CRT_SECURE_NO_WARNINGS 1

#include <lib/pal/pal.h>
#include <lib/gfx/gfx.h>
#include <algorithm>

#pragma push_macro("free")
#undef free
#include <opencv2/opencv.hpp>
#pragma pop_macro("free")

#ifdef _MSC_VER
#pragma warning(disable : 4503)
#endif
#include <nlohmann-json/json.hpp>

#include <xo/xo/xo.h>
