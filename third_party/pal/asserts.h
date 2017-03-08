#pragma once

#include "defs.h"

namespace imqs {
IMQS_PAL_API void IMQS_NORETURN Die(const char* file, int line, const char* msg);

#define IMQS_DIE() Die(__FILE__, __LINE__, "")
#define IMQS_DIE_MSG(msg) Die(__FILE__, __LINE__, msg)

// NOTE: This is compiled in all builds (Debug, Release)
#define IMQS_ASSERT(f) (void) ((f) || (Die(__FILE__, __LINE__, #f), 0))
}