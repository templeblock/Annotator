#include "pch.h"
#include "zip.h"
#include "../alloc.h"
#include "../io/io.h"
#include "../os/os.h"

using namespace std;

namespace imqs {
namespace archive {

static Error MakeMzErr(int32_t e, const char* action) {
	return Error::Fmt("Error %v: %v", action, e);
}

ZipFile::ZipFile() {
}

ZipFile::~ZipFile() {
	Close();
}

void ZipFile::Close() {
	mz_zip_close(MzZip);
	mz_stream_mem_delete(&MzStream);
	free(MemBuf);
	MzZip      = nullptr;
	MzStream   = nullptr;
	MemBuf     = nullptr;
	MemBufSize = 0;
	Filenames.clear();
}

Error ZipFile::OpenMem(const void* data, size_t len) {
	Close();

	// mz_stream_mem_set_buffer takes an int32
	if ((uint64_t) len >= (uint64_t) 0x7fffffff)
		return Error::Fmt("Zip file is too large to open in memory (%v bytes)", len);

	MemBuf = imqs_malloc_or_die(len);
	memcpy(MemBuf, data, len);
	MemBufSize = len;

	// fill zip_buffer with zip contents
	mz_stream_mem_create(&MzStream);
	mz_stream_mem_set_buffer(MzStream, MemBuf, (int32_t) MemBufSize);
	auto err = mz_stream_open(MzStream, NULL, MZ_OPEN_MODE_READ);
	if (err != MZ_OK)
		return MakeMzErr(err, "opening zip stream");

	MzZip = mz_zip_open(MzStream, MZ_OPEN_MODE_READ);
	if (!MzZip)
		return MakeMzErr(0, "opening zip");

	err = mz_zip_goto_first_entry(MzZip);
	if (err != MZ_OK)
		return MakeMzErr(err, "seeking to first file in zip");
	while (true) {
		err = mz_zip_entry_read_open(MzZip, 0, nullptr);
		if (err != MZ_OK)
			return MakeMzErr(err, "opening zip file entry");

		mz_zip_file* info = nullptr;
		err               = mz_zip_entry_get_info(MzZip, &info);
		if (err != MZ_OK)
			return MakeMzErr(err, "reading zip info");

		// 7-zip saves directories, but Windows built-in compressor doesn't
		bool isDir = !!(info->external_fa & 16) && !(info->external_fa & 32) && info->filename[info->filename_size - 1] == '/';
		if (!isDir)
			Filenames.push_back(string(info->filename, info->filename_size));

		err = mz_zip_entry_close(MzZip);
		if (err != MZ_OK)
			return MakeMzErr(err, "closing zip entry");

		err = mz_zip_goto_next_entry(MzZip);
		if (err == MZ_END_OF_LIST)
			break;
		else if (err != MZ_OK)
			return MakeMzErr(err, "seeking to next file in zip");
	}

	return Error();
}

static int32_t CompareFilename(void* handle, const char* filename1, const char* filename2) {
	return strcmp(filename1, filename2);
}

Error ZipFile::ReadWholeFile(const std::string& name, std::string& content) {
	int err = mz_zip_locate_entry(MzZip, name.c_str(), CompareFilename);
	if (err == MZ_END_OF_LIST)
		return os::ErrENOENT;

	err = mz_zip_entry_read_open(MzZip, 0, nullptr);
	if (err != MZ_OK)
		return MakeMzErr(err, "opening zip entry");

	char buf[1024];
	while (true) {
		int n = mz_zip_entry_read(MzZip, buf, sizeof(buf));
		if (n < 0)
			return MakeMzErr(n, "reading zip entry contents");
		if (n == 0)
			break;
		content.append(buf, n);
	}

	err = mz_zip_entry_close(MzZip);
	if (err != MZ_OK)
		return MakeMzErr(err, "closing zip entry");

	return Error();
}

} // namespace archive
} // namespace imqs
