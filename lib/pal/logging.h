#pragma once

#include <third_party/uberlog/uberlog.h>

namespace imqs {
namespace logging {

// Initialize a standard IMQS logger. The 'module' specified here becomes the filename inside imqsvar/logs/<module>.log
IMQS_PAL_API void        CreateLogger(const char* module, uberlog::Logger& logger);
IMQS_PAL_API void        SetupCrashHandler(const char* appName);
IMQS_PAL_API void        SetCrashLogger(uberlog::Logger* logger);
IMQS_PAL_API const char* CrashDumpDir();

class ITeeTarget {
public:
	virtual void AppendLog(uberlog::Level level, const std::string& message) = 0;
};

// A logger that "T"'s off the log messages so that they go to uberlog, as well as
// another interface of your choice. The log messages that are send to the Tee interface
// do not have the standard uberlog prefix of date, level, threadid; The tee gets
// only the pure log message and the level.
class IMQS_PAL_API LogTee {
public:
	ITeeTarget*     Tee = nullptr;
	uberlog::Logger Uber;

	template <typename... Args>
	void Log(uberlog::Level level, const char* format_str, const Args&... args) {
		if (level < Uber.GetLevel())
			return;
		Uber.Log(level, format_str, args...);
		if (Tee)
			Tee->AppendLog(level, tsf::fmt(format_str, args...));
	}

	template <typename... Args>
	void Debug(const char* format_str, const Args&... args) { Log(uberlog::Level::Debug, format_str, args...); }

	template <typename... Args>
	void Info(const char* format_str, const Args&... args) { Log(uberlog::Level::Info, format_str, args...); }

	template <typename... Args>
	void Warn(const char* format_str, const Args&... args) { Log(uberlog::Level::Warn, format_str, args...); }

	template <typename... Args>
	void Error(const char* format_str, const Args&... args) { Log(uberlog::Level::Error, format_str, args...); }

	template <typename... Args>
	void Fatal(const char* format_str, const Args&... args) { Log(uberlog::Level::Fatal, format_str, args...); }
};

} // namespace logging
} // namespace imqs
