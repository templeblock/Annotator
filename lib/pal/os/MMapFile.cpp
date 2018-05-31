#include "pch.h"
#include "MMapFile.h"
#include "../alloc.h"
#include "../strings/utf.h"
#include "os.h"
#include <fcntl.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

namespace imqs {
namespace os {

MMapFile::MMapFile() {
}

MMapFile::~MMapFile() {
	Close();
}

Error MMapFile::Open(const std::string& filename) {
	return OpenInternal(filename, false);
}

Error MMapFile::Create(const std::string& filename) {
	return OpenInternal(filename, true);
}

Error MMapFile::OpenInternal(const std::string& filename, bool create) {
	Close();
	Filename = filename;
	IsWrite  = create;
#ifdef _WIN32
	DWORD access = create ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ;
	DWORD share  = create ? 0 : FILE_SHARE_READ;
	DWORD crd    = create ? CREATE_ALWAYS : OPEN_EXISTING;
	HFile        = CreateFileW(towide(filename).c_str(), access, share, nullptr, crd, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (HFile == INVALID_HANDLE_VALUE)
		return ErrorFrom_GetLastError();
#else
	int flags = create ? O_RDWR | O_CREAT | O_TRUNC : O_RDONLY;
	FD        = open(filename.c_str(), flags, 0660);
	if (FD == -1)
		return ErrorFrom_errno();
#endif
	auto err = RecreateMapping();
	if (!err.OK()) {
		Close();
		return err;
	}
	return Error();
}

Error MMapFile::RecreateMapping(int64_t minSize) {
	int64_t newWriteSize = MappedSize;
	if (IsWrite) {
		while (true) {
			uint64_t growth = std::min<uint64_t>(newWriteSize / 2, 64 * 1024 * 1024);
			growth          = std::max<uint64_t>(growth, 65536);
			newWriteSize += growth;
			if (minSize == 0 || newWriteSize >= minSize)
				break;
		}
	}

#ifdef _WIN32
	if (Base) {
		if (!UnmapViewOfFile(Base))
			return Error::Fmt("Failed to unmap file view %v: %v", Filename, ErrorFrom_GetLastError().Message());
		Base = nullptr;
	}
	if (HMapping) {
		if (!CloseHandle(HMapping))
			return Error::Fmt("Failed to close file mapping %v: %v", Filename, ErrorFrom_GetLastError().Message());
		HMapping = nullptr;
	}
	DWORD protect = IsWrite ? PAGE_READWRITE : PAGE_READONLY;
	if (IsWrite) {
		MappedSize = newWriteSize;
	} else {
		BY_HANDLE_FILE_INFORMATION inf;
		if (!GetFileInformationByHandle(HFile, &inf))
			return ErrorFrom_GetLastError();
		CachedLength = (uint64_t) inf.nFileSizeHigh << 32 | inf.nFileSizeLow;
		if (CachedLength == 0)
			return Error::Fmt("Cannot memory map an empty file: %v", Filename);
		MappedSize = CachedLength;
	}
	uint64_t size = (uint64_t) MappedSize;
	HMapping      = CreateFileMappingW(HFile, nullptr, protect, size >> 32, size & 0xffffffff, nullptr);
	if (HMapping == nullptr)
		return ErrorFrom_GetLastError();
	DWORD access = IsWrite ? FILE_MAP_READ | FILE_MAP_WRITE : FILE_MAP_READ;
	Base         = (uint8_t*) MapViewOfFile(HMapping, access, 0, 0, 0);
	if (!Base)
		return Error::Fmt("Failed to map %v bytes of file %v: %v", MappedSize, Filename, ErrorFrom_GetLastError().Message());
#else
	if (Base) {
		int e = munmap(Base, MappedSize);
		if (e != 0)
			return Error::Fmt("Failed to unmap %v: %v", Filename, ErrorFrom_errno().Message());
		Base       = nullptr;
		MappedSize = 0;
	}
	if (IsWrite) {
		MappedSize = newWriteSize;
		int e      = ftruncate(FD, MappedSize);
		if (e != 0)
			return Error::Fmt("Failed to fruncate mmap file %v: %v", Filename, ErrorFrom_errno().Message());
	} else {
		off_t r = lseek(FD, 0, SEEK_END);
		if (r == -1)
			return Error::Fmt("Unable to determine file length of %v: %v", Filename, ErrorFrom_errno().Message());
		if (r == 0)
			return Error::Fmt("Cannot memory map an empty file: %v", Filename);
		MappedSize   = r;
		CachedLength = r;
	}
	int prot = IsWrite ? PROT_READ | PROT_WRITE : PROT_READ;
	Base     = (uint8_t*) mmap(nullptr, (size_t) MappedSize, prot, MAP_SHARED, FD, 0);
	if (!Base) {
		auto err     = Error::Fmt("Failed to map %v bytes of file %v: %v", MappedSize, Filename, ErrorFrom_errno().Message());
		MappedSize   = 0;
		CachedLength = 0;
		return err;
	}
#endif
	return Error();
}

Error MMapFile::Write(const void* buf, size_t len) {
	auto err = WriteAt(Pos, buf, len);
	if (!err.OK())
		return err;
	Pos += len;
	return Error();
}

Error MMapFile::Read(void* buf, size_t& len) {
	auto err = ReadAt(Pos, buf, len);
	if (!err.OK())
		return err;
	Pos += len;
	return Error();
}

Error MMapFile::ReadExactly(void* buf, size_t len) {
	size_t actual = len;
	auto   err    = Read(buf, actual);
	if (actual != len)
		return ErrEOF;
	return Error();
}

Error MMapFile::ReadExactlyAt(int64_t pos, void* buf, size_t len) {
	size_t actual = len;
	ReadAt(pos, buf, len);
	if (actual != len)
		return ErrEOF;
	return Error();
}

Error MMapFile::ReadAt(int64_t pos, void* buf, size_t& len) {
	int64_t maxLen = CachedLength - pos;
	if (maxLen <= 0) {
		len = 0;
		return ErrEOF;
	}
	IMQS_ASSERT(maxLen < (uint64_t)(size_t) -1);
	len = std::min<size_t>(len, (size_t) maxLen);
	memcpy(buf, Base + pos, len);
	return Error();
}

Error MMapFile::WriteAt(int64_t pos, const void* buf, size_t len) {
	if ((int64_t) len < 0)
		return Error::Fmt("Invalid write of %v bytes", len);
	if (pos + (int64_t) len > MappedSize) {
		auto err = RecreateMapping(pos + len);
		if (!err.OK())
			return err;
	}
	memcpy(Base + pos, buf, len);
	CachedLength = std::max(CachedLength, pos + (int64_t) len);
	return Error();
}

Error MMapFile::SeekWithResult(int64_t offset, io::SeekWhence whence, int64_t& newPosition) {
	switch (whence) {
	case io::SeekWhence::Begin: Pos = offset; break;
	case io::SeekWhence::Current: Pos += offset; break;
	case io::SeekWhence::End: Pos = CachedLength + offset; break;
	}
	newPosition = Pos;
	return Error();
}

int64_t MMapFile::Length() {
	return CachedLength;
}

bool MMapFile::IsOpen() const {
#ifdef _WIN32
	return HFile != INVALID_HANDLE_VALUE;
#else
	return FD != -1;
#endif
}

Error MMapFile::Close() {
	Error err;
#ifdef _WIN32
	if (Base)
		UnmapViewOfFile(Base);
	if (HMapping)
		CloseHandle(HMapping);
	if (HFile != INVALID_HANDLE_VALUE) {
		if (IsWrite) {
			LARGE_INTEGER pos;
			pos.QuadPart = (LONGLONG) CachedLength;
			LARGE_INTEGER npos;
			if (!SetFilePointerEx(HFile, pos, &npos, FILE_BEGIN))
				err = Error::Fmt("Failed to seek %v to position %v: %v", Filename, CachedLength, ErrorFrom_GetLastError().Message());
			if (err.OK()) {
				if (!SetEndOfFile(HFile))
					err = Error::Fmt("Failed to truncate file %v to %v: %v", Filename, CachedLength, ErrorFrom_GetLastError().Message());
			}
		}
		CloseHandle(HFile);
	}
	HFile    = INVALID_HANDLE_VALUE;
	HMapping = nullptr;
#else
	if (Base)
		munmap(Base, MappedSize);
	if (FD != -1) {
		int e      = ftruncate(FD, CachedLength);
		int _errno = errno;
		close(FD);
		if (e == -1 && _errno == EINVAL) {
			// apparently this is expected: https://stackoverflow.com/questions/20320742/ftruncate-failed-at-the-second-time
			e = 0;
		}
		if (e != 0)
			err = Error::Fmt("Failed to truncate file %v to %v: %v", Filename, CachedLength, ErrorFrom_errno(_errno).Message());
	}
	FD = -1;
#endif
	Pos          = 0;
	CachedLength = 0;
	MappedSize   = 0;
	Base         = nullptr;
	Filename     = "";
	return err;
}

} // namespace os
} // namespace imqs
