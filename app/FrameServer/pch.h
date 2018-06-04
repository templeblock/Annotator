#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#define _CRT_SECURE_NO_WARNINGS 1

#include <algorithm>

#include <lib/pal/pal.h>
#include <lib/Video/Video.h>
#include <lib/Train/Train.h>
#include <lib/gfx/gfx.h>

#include <phttp/phttp.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4503)
#endif
#include <nlohmann-json/json.hpp>
