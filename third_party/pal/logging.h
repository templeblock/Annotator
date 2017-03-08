#pragma once

#include <third_party/uberlog/uberlog.h>

namespace imqs {
namespace logging {

// Initialize a standard IMQS logger. The 'module' specified here becomes the filename inside imqsvar/logs/<module>.log
IMQS_PAL_API void        CreateLogger(const char* module, uberlog::Logger& logger);
IMQS_PAL_API void        SetupCrashHandler(const char* appName);
IMQS_PAL_API const char* CrashDumpDir();
}
}
