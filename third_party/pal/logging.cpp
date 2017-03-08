#include "pch.h"
#include "logging.h"

#ifdef _WIN32
#include <Dbghelp.h>
#endif

namespace imqs {
namespace logging {

IMQS_PAL_API void CreateLogger(const char* module, uberlog::Logger& logger) {
#ifdef _WIN32
	std::string logRootPath = "c:/imqsvar/logs/";
#else
	std::string logRootPath = "/var/log/imqs/";
#endif

	/*
	auto sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(logRootPath + module, SPDLOG_FILENAME_T("log"), 10 * 1024 * 1024, 3);
	// Disable libc buffering of logs. This is bad for performance, because you get a kernel Write() call with every
	// log message. However, we do not want to lose a log message, if that log message is immediately succeeded by
	// a crash. A better solution is to have a child process which is doing the writes, and we communicate with that
	// child process via shared memory. Until we have such a system available, we do this.
	// spdlog also has an "asynchronous" mode, but that suffers from the same buffering problem.
	setvbuf(sink->file_handle(), nullptr, _IONBF, 0);
	auto log = std::make_shared<spdlog::logger>(module, sink);
	log->set_pattern("%Y-%m-%dT%H:%M:%S.%f%z [%L] %t %v");
	log->set_level(spdlog::level::info);
	log->flush_on(spdlog::level::off);
	return log;
	*/

	// Uberlog solves the above problem, and does so without any cost in performance.
	logger.SetArchiveSettings(30 * 1024 * 1024, 3);
	logger.Open((logRootPath + module + ".log").c_str());
}

#ifdef _WIN32

namespace exception_handler {

static const size_t MaxModuleNameLength   = 30;
static const size_t MaxArchivedCrashDumps = 10;
static bool         IsBusy                = false;
static char         ModuleName[MaxModuleNameLength];

// We need to be careful during exception handling, not to do too much, because we
// risk causing a double exception.
// sprintf, for the case we use here, does not allocate memory. MSVC 2015 x64 Update 3.

static void NextCrashDumpFilename(char (&name)[256]) {
	char     oldest[256]  = {0};
	uint64_t oldest_mtime = UINT64_MAX;

	for (int i = 0; i < MaxArchivedCrashDumps; i++) {
		sprintf(name, "%s\\%s-%d.mdmp", CrashDumpDir(), ModuleName, i + 1);
		WIN32_FILE_ATTRIBUTE_DATA data;
		DWORD                     res = GetFileAttributesExA(name, GetFileExInfoStandard, &data);
		if (res == 0) {
			// assume an error means the file does not exist
			return;
		}
		uint64_t mtime = ((uint64_t) data.ftLastWriteTime.dwHighDateTime << 32) | (uint64_t) data.ftLastWriteTime.dwLowDateTime;
		if (mtime <= oldest_mtime) {
			oldest_mtime = mtime;
			strcpy(oldest, name);
		}
	}

	strcpy(name, oldest);
}

static LONG WINAPI ExceptionHandler(EXCEPTION_POINTERS* exPointers) {
	if (IsBusy)
		return EXCEPTION_EXECUTE_HANDLER;
	IsBusy = true;

	char dumpFilename[256];
	NextCrashDumpFilename(dumpFilename);
	dumpFilename[arraysize(dumpFilename) - 1] = 0;

	// First location searched for by CreateProcess is the path of the current executable
	char crashHelperExe[512] = {0};
	strcat(crashHelperExe, "CrashHelper.exe ");
	strcat(crashHelperExe, ModuleName);
	strcat(crashHelperExe, " ");
	strcat(crashHelperExe, dumpFilename);

	CreateDirectoryA(CrashDumpDir(), NULL);

	HANDLE hFile = CreateFileA(dumpFilename, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile != INVALID_HANDLE_VALUE) {
		MINIDUMP_EXCEPTION_INFORMATION mExInfo;
		mExInfo.ThreadId          = GetCurrentThreadId();
		mExInfo.ExceptionPointers = exPointers;
		mExInfo.ClientPointers    = TRUE;
		// It's tempting to try things such as MiniDumpWithIndirectlyReferencedMemory, but until I can get some real-world statistics,
		// I'm too afraid to include it, for fear that the size might be too large.
		MINIDUMP_TYPE type    = MiniDumpNormal;
		bool          writeOK = MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, type, &mExInfo, NULL, NULL) == TRUE;
		CloseHandle(hFile);
		if (writeOK) {
			// Launch a child process that will send the crash dump over the network
			STARTUPINFO si;
			memset(&si, 0, sizeof(si));
			si.cb = sizeof(si);
			PROCESS_INFORMATION pi;
			bool                childOK = CreateProcessA(NULL, crashHelperExe, NULL, NULL, false, 0, NULL, NULL, &si, &pi) == TRUE;
			if (!childOK)
				printf("Unable to launch CrashHelper.exe\n");
			CloseHandle(pi.hProcess);
			CloseHandle(pi.hThread);
		}
	}

	IsBusy = false;
	return EXCEPTION_EXECUTE_HANDLER;
}
}

IMQS_PAL_API void SetupCrashHandler(const char* appName) {
	strncpy(exception_handler::ModuleName, appName, arraysize(exception_handler::ModuleName));
	exception_handler::ModuleName[arraysize(exception_handler::ModuleName) - 1] = 0;

	SetUnhandledExceptionFilter(&exception_handler::ExceptionHandler);
}

IMQS_PAL_API const char* CrashDumpDir() {
	return "c:\\imqsvar\\crashdumps";
}

#else

IMQS_PAL_API void SetupCrashHandler(const char* appName) {
}

IMQS_PAL_API const char* CrashDumpDir() {
	return "/var/log/imqs/crashdumps";
}
#endif
}
}
