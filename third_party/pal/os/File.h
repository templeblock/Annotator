#pragma once

#include "../io/io.h"

namespace imqs {
namespace os {

/* File.
*/
class IMQS_PAL_API File : public io::Reader, public io::Writer, public io::Seeker {
public:
	File();
	~File();

	Error Write(const void* buf, size_t len) override;
	Error Read(void* buf, size_t& len) override;
	Error SeekWithResult(int64_t offset, io::SeekWhence whence, int64_t& newPosition) override;

	// Open a file for read-only access
	Error Open(const std::string& filename);

	// Create a new file, or truncate an existing file
	Error Create(const std::string& filename);

	void Close();

private:
#ifdef _WIN32
	HANDLE FH = INVALID_HANDLE_VALUE;
	Error  WinCreateFile(const std::string& filename, DWORD desiredAccess, DWORD shareMode, DWORD creationDisposition, DWORD flagsAndAttributes);
#else
	int FD = -1;
#endif
};

} // namespace os
} // namespace imqs
