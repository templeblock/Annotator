#include "pch.h"
#include "File.h"
#include "os.h"

#include <fcntl.h>

namespace imqs {
namespace os {

File::File() {
}

File::~File() {
	Close();
}

Error File::Write(const void* buf, size_t len) {
	while (true) {
		ssize_t r = write(FD, buf, len);
		if (r == -1)
			return ErrorFrom_errno();
		len -= r;
		(char*&) buf += r;
		if (len == 0)
			return Error();
	}

	// unreachable
	return Error();
}

Error File::Read(void* buf, size_t& len) {
	ssize_t r = read(FD, buf, len);
	if (r == -1)
		return ErrorFrom_errno();
	len = r;
	return Error();
}

Error File::Open(const std::string& filename) {
	int r = open(filename.c_str(), O_RDONLY);
	if (r == -1)
		return ErrorFrom_errno();
	FD = r;
	return Error();
}

Error File::Create(const std::string& filename) {
	int r = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
	if (r == -1)
		return ErrorFrom_errno();
	FD = r;
	return Error();
}

Error File::Seek(int64_t offset, SeekWhence whence, int64_t& newPosition) {
	int m = 0;
	switch (whence) {
	case SeekWhence::Begin: m = SEEK_SET; break;
	case SeekWhence::Current: m = SEEK_CUR; break;
	case SeekWhence::End: m = SEEK_END; break;
	}
	off_t r = lseek(FD, offset, m);
	if (r == -1)
		return ErrorFrom_errno();
	newPosition = r;
	return Error();
}

void File::Close() {
	if (FD != -1)
		close(FD);
	FD = -1;
}
}
}
