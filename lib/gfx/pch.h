#pragma once
#include "common.h"

#include <png.h> // On linux, this goes to mapnik deps mason_packages

// implementation is inside stb.cpp
#include <third_party/stb/stb_image.h>

// intrinsics for SSE,AVX,etc
#include <x86intrin.h>

// fast integer divides by constants
#define LIBDIVIDE_USE_SSE2
#define LIBDIVIDE_USE_SSE4_1
#include <third_party/libdivide/libdivide.h>
